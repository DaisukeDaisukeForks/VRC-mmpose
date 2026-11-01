# Third-party notices
## InstantHMR (main application)
Distribution: https://huggingface.co/momolesang/InstantHMR
Revision: 3504446fc31e7f76fdb1cd7e463189e7cf0fdd0f. SHA-256: f717558094f57d7c9cd084d7981874dbcc3f08e7308501eb98c12844e2cce3b7.
Source: https://github.com/Naturalpad-studio/InstantHMR (Apache-2.0).
Model weights: SAM License as specified by the model distributor: https://huggingface.co/momolesang/InstantHMR. Weights are directly fetched from the original public host, not included in this repository or ZIP. No dense MHR mesh decoder or body assets are bundled.
The MHR70 names and connectivity adapter follow the upstream skeleton. Original model coordinates are metres, Y-down; display changes are described in README.md.
Project reference: https://github.com/open-mmlab/mmpose
## RTMW3D-X
Publisher: OpenMMLab; ONNX distribution: Soykaf/RTMW3D-x.
Source: https://huggingface.co/Soykaf/RTMW3D-x
Revision: a4f0d54c52ee44ee4e1dda1936d714490c6234d2.
License: Apache-2.0 as declared by the distribution. Model weights are fetched from the original host and are not included in this repository or source ZIP.
Official preprocessing/decoding: https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d
## RTMDet-s
Distribution: https://huggingface.co/ziq/rtm
Revision: 6bdd0110307167f5f003e92e34099a21bd61283c.
License: MIT as declared by that distribution. Model weights are fetched from the original host, not included here. Underlying RTMDet implementation: OpenMMLab MMDetection, Apache-2.0.
## ONNX Runtime Web
https://github.com/microsoft/onnxruntime
Copyright Microsoft Corporation. License: MIT. Main application: version 1.27.0 JSEP bundle fetched from jsDelivr, hash-verified and modified in memory using the user's supplied FP32-accumulation patch in dist/ort-patch.js. Runtime files are not vendored. Existing comparison code and old fixtures retain their original runtime metadata.
## Sample video
`dist/sample.mp4` is the first 8 seconds of OpenCV `samples/data/vtest.avi`, converted to H.264 for browser decoding.
Original: https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi
OpenCV distribution license: https://github.com/opencv/opencv/blob/master/LICENSE (Apache-2.0). This fixture demonstrates person detection and is not a 60fps performance or tracking-accuracy benchmark. Original metadata: 768×576, 10fps. No generated or synthetic human pose results are used.
## Fonts
DM Sans: https://fonts.google.com/specimen/DM+Sans
Noto Sans JP: https://fonts.google.com/noto/specimen/Noto+Sans+JP
SIL Open Font License, loaded from Google Fonts. System sans-serif is used when fonts are unavailable.
