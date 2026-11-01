"""Run the pinned, unmodified public InstantHMR ONNX on GitHub Actions only."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REVISION = '3504446fc31e7f76fdb1cd7e463189e7cf0fdd0f'
SHA256 = 'f717558094f57d7c9cd084d7981874dbcc3f08e7308501eb98c12844e2cce3b7'
URL = f'https://huggingface.co/momolesang/InstantHMR/resolve/{REVISION}/instanthmr.onnx'
CACHE = ROOT / '.model-cache'
CACHE.mkdir(exist_ok=True)
path = CACHE / 'instanthmr.onnx'
if not path.exists():
    urllib.request.urlretrieve(URL, path)
assert path.stat().st_size == 80961622
with path.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == SHA256
graph = onnx.load(path)
onnx.checker.check_model(graph)
metadata = {p.key: p.value for p in graph.metadata_props}
options = ort.SessionOptions()
options.intra_op_num_threads = 4
session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
inputs = {v.name: {'shape': v.shape, 'type': v.type} for v in session.get_inputs()}
outputs = {v.name: {'shape': v.shape, 'type': v.type} for v in session.get_outputs()}
assert set(inputs) == {'image', 'cliff_cond'}, inputs
assert inputs['image']['shape'][1:] == [3, 224, 224], inputs
assert inputs['cliff_cond']['shape'][1:] == [3], inputs
expected = {'mhr_params': [204], 'shape_params': [45], 'cam_trans': [3], 'joints_2d': [70, 2], 'joints_3d': [70, 3]}
assert set(outputs) == set(expected), outputs
for name, dims in expected.items():
    assert outputs[name]['shape'][1:] == dims, (name, outputs[name])
assert str(metadata.get('cliff_focal', '')).lower() != 'true', 'Pinned preprocessing contract changed'
report = {'repo': 'momolesang/InstantHMR', 'revision': REVISION, 'sha256': SHA256,
          'bytes': path.stat().st_size, 'runtime': ort.__version__, 'provider': 'CPUExecutionProvider',
          'inputs': inputs, 'outputs': outputs, 'metadata': metadata,
          'opsets': {v.domain or 'ai.onnx': v.version for v in graph.opset_import},
          'operators': dict(collections.Counter(v.op_type for v in graph.graph.node)),
          'initializerTypes': dict(collections.Counter(onnx.TensorProto.DataType.Name(v.data_type) for v in graph.graph.initializer)),
          'frames': [], 'notes': ['CPU contract and numerical reference, not a WebGPU speed or GPU accuracy result']}
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float64)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float64)
dtype = np.float16 if inputs['image']['type'] == 'tensor(float16)' else np.float32
cond_dtype = np.float16 if inputs['cliff_cond']['type'] == 'tensor(float16)' else np.float32
def infer(image, cond):
    started = time.perf_counter()
    result = dict(zip(outputs, session.run(list(outputs), {'image': image.astype(dtype), 'cliff_cond': np.array([cond], dtype=cond_dtype)})))
    assert all(np.isfinite(value).all() for value in result.values()), 'Non-finite model output'
    return result, (time.perf_counter() - started) * 1000

yy, xx = np.indices((224, 224))
probe = np.stack([((xx + 3 * yy + 17 * c) % 256 - MEAN[c]) / STD[c] for c in range(3)])[None].astype(np.float32)
reference, _ = infer(probe, [0, 0, 1])
probe_reference = {'schema': 'instanthmr-probe/v1', 'modelSha256': SHA256, 'inputs': inputs, 'outputs': outputs,
                   'cliffFocal': False, 'values': {k: v.astype(np.float32).reshape(-1).tolist() for k, v in reference.items()},
                   'tolerance': {'joints_2d': 0.025, 'joints_3d': 0.025, 'cam_trans': 0.05},
                   'notes': 'Actual pinned ONNX CPU output for deterministic 224px RGB probe; startup checks GPU numerical agreement.'}
for timestamp in [0, 2, 5]:
    png = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(timestamp), '-i', str(ROOT / 'dist/sample.mp4'), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'])
    image = Image.open(io.BytesIO(png)).convert('RGB')
    w, h = image.size
    side = max(w, h) * 1.2
    x, y = (w - side) / 2, (h - side) / 2
    patch = image.transform((224, 224), Image.Transform.AFFINE, (side / 224, 0, x, 0, side / 224, y), Image.Resampling.BILINEAR)
    pixels = np.asarray(patch, dtype=np.float64)
    inp = ((pixels - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    result, elapsed = infer(inp, [0, 0, 1])
    xy = result['joints_2d'][0].astype(np.float64)
    xy = (xy + 1) * side / 2 + np.array([x, y])
    xyz = result['joints_3d'][0].astype(np.float64)
    root = xyz[[9, 10]].mean(axis=0)
    report['frames'].append({'timeSeconds': timestamp, 'poseMs': elapsed, 'finiteJoints': 70,
                             'ankles3d': xyz[[13, 14]].tolist(), 'feet3d': xyz[[15, 16, 17, 18, 19, 20]].tolist(),
                             'cameraPrediction': result['cam_trans'][0].astype(float).tolist()})
    if timestamp == 0:
        fixture = {'roi': {'x': x, 'y': y, 'width': side, 'height': side}, 'source': [w, h],
                   'outputs': {k: {'dims': list(v.shape), 'type': 'float32', 'data': v.astype(np.float32).reshape(-1).tolist()} for k, v in result.items()},
                   'expectedXY': xy.tolist(), 'expectedRelative3d': (xyz - root).tolist()}
outdir = ROOT / 'verification' / 'instant'
outdir.mkdir(parents=True, exist_ok=True)
(outdir / 'cpu-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(outdir / 'decoder-fixture.json').write_text(json.dumps(fixture), encoding='utf-8')
(outdir / 'probe-reference.json').write_text(json.dumps(probe_reference), encoding='utf-8')
print('INSTANT_CONTRACT=' + json.dumps(report))
print('INSTANT_PROBE=' + json.dumps(probe_reference))
print('INSTANT_FIXTURE=' + json.dumps(fixture))
