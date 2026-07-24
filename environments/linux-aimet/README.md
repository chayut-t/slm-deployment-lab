# Linux AIMET environment

Use the common repository lock for offline tooling. T40 owns the tested AIMET
extension and must pin exact Linux, Python, PyTorch, Transformers, ONNX,
AIMET, and applicable Qualcomm package versions after its calibration smoke
test. T30 separately records public AI Hub client and hosted compiler/runtime
versions.

Do not assume that a locally resolvable wheel establishes QAIRT, target, or
hosted-runtime compatibility.
