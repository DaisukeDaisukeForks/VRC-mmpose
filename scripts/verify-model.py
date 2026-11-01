"""Verify downloaded original models and run the same video->ROI->SimCC contract on CPU.
This is a verification utility, not an alternative application inference backend.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('MODEL_CACHE_DIR', ROOT / '.model-cache'))
CACHE.mkdir(parents=True, exist_ok=True)
MODELS = {
    'pose': ('rtmw3d-x.onnx', 'https://huggingface.co/Soykaf/RTMW3D-x/resolve/a4f0d54c52ee44ee4e1dda1936d714490c6234d2/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx', '4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd'),
    'detector': ('rtmdet-s.onnx', 'https://huggingface.co/ziq/rtm/resolve/6bdd0110307167f5f003e92e34099a21bd61283c/rtmdet-s.onnx', '3bb7246a8b5104dd94e9f9bbfcf92d2cd9e55d344a9a5e8b7b84ebcfeedafe14'),
}
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

def model_path(key):
    name, url, expected = MODELS[key]
    target = CACHE / name
    if not target.exists():
        urllib.request.urlretrieve(url, target)
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    assert digest == expected, (name, digest)
    return target

options = ort.SessionOptions()
options.intra_op_num_threads = 4
pose = ort.InferenceSession(str(model_path('pose')), sess_options=options, providers=['CPUExecutionProvider'])
detector = ort.InferenceSession(str(model_path('detector')), sess_options=options, providers=['CPUExecutionProvider'])
assert pose.get_inputs()[0].shape == [1, 3, 384, 288]
assert [out.shape for out in pose.get_outputs()] == [[1, 133, 576], [1, 133, 768], [1, 133, 576]]
summary = {'model': 'Soykaf/RTMW3D-x', 'sha256': MODELS['pose'][2], 'backend': 'CPUExecutionProvider',
           'inputs': [(x.name, x.shape) for x in pose.get_inputs()], 'outputs': [(x.name, x.shape) for x in pose.get_outputs()], 'frames': []}
for timestamp in [0, 2, 5]:
    png = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(timestamp), '-i', str(ROOT / 'dist/sample.mp4'), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'])
    image = Image.open(io.BytesIO(png)).convert('RGB')
    w, h = image.size
    scale = min(640 / w, 640 / h)
    resized = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)
    letterbox = Image.new('RGB', (640, 640), (114, 114, 114))
    letterbox.paste(resized, (0, 0))
    bgr = np.asarray(letterbox, dtype=np.float32)[..., ::-1]
    det_input = ((bgr - MEAN[::-1]) / STD[::-1]).transpose(2, 0, 1)[None].copy()
    det_out = detector.run(None, {detector.get_inputs()[0].name: det_input})
    dets, labels = det_out[0].reshape(-1, 5), det_out[1].reshape(-1)
    persons = [row for row, label in zip(dets, labels) if int(label) == 0 and row[4] > .35]
    assert persons, f'No person detected at {timestamp}s'
    box = max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4] / scale
    box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
    cx, cy = (box[:2] + box[2:]) / 2
    sw, sh = (box[2:] - box[:2]) * 1.25
    if sw / sh > .75:
        sh = sw / .75
    else:
        sw = sh * .75
    x, y = float(cx - sw / 2), float(cy - sh / 2)
    # PIL inverse affine maps target pixel centres into the original full-resolution frame.
    patch = image.transform((288, 384), Image.Transform.AFFINE, (float(sw / 288), 0, x, 0, float(sh / 384), y), Image.Resampling.BILINEAR)
    inp = ((np.asarray(patch, dtype=np.float32) - MEAN) / STD).transpose(2, 0, 1)[None].copy()
    started = time.perf_counter()
    outputs = pose.run(None, {pose.get_inputs()[0].name: inp})
    elapsed = (time.perf_counter() - started) * 1000
    assert all(np.isfinite(a).all() for a in outputs)
    indices = np.stack([a[0].argmax(-1) for a in outputs], -1)
    scores = np.minimum(outputs[0][0].max(-1), outputs[1][0].max(-1))
    decoded = indices.astype(np.float64) / 2
    decoded[:, 0] = x + decoded[:, 0] / 288 * sw
    decoded[:, 1] = y + decoded[:, 1] / 384 * sh
    decoded[:, 2] = (decoded[:, 2] / 144 - 1) * 2.1744869
    assert int((scores > .3).sum()) >= 15, f'Insufficient observed joints: {scores}'
    summary['frames'].append({'timeSeconds': timestamp, 'persons': len(persons), 'validJoints': int((scores > .3).sum()), 'poseMs': elapsed,
                              'hipRelativeZ': decoded[[11, 12], 2].tolist(), 'roi': [x, y, float(sw), float(sh)]})
    if timestamp == 0:
        fixture = {'shape': [a.shape for a in outputs], 'roi': {'x': x, 'y': y, 'width': float(sw), 'height': float(sh)}, 'source': [w, h],
                   'indices': indices.tolist(), 'scores': scores.tolist(), 'decoded': decoded.tolist()}
        fixture_path = os.environ.get('FIXTURE_PATH')
        if fixture_path:
            Path(fixture_path).write_text(json.dumps(fixture), encoding='utf-8')
print(json.dumps(summary, indent=2))
output = os.environ.get('VERIFICATION_OUTPUT')
if output:
    Path(output).write_text(json.dumps(summary, indent=2), encoding='utf-8')
