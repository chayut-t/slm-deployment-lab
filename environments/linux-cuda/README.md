# Linux CUDA environment

Use the common repository lock for offline tooling. T60 owns the tested Linux
CUDA extension and must pin the exact distribution, kernel, NVIDIA driver,
CUDA, cuDNN, GPU, ONNX Runtime GPU, PyTorch, Transformers, and ONNX versions
observed in its selected environment.

Do not infer CUDA/driver compatibility from package metadata alone, and do not
record a Colab or Kaggle image tag such as `latest` as a version. T02 records
available free GPU access; launching a paid fallback requires explicit user
approval.
