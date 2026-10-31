# CPU-only Triton server with Python-backend extras for the Qwen generator.
# The stock NGC image already includes onnxruntime + python backends.
FROM nvcr.io/nvidia/tritonserver:24.12-py3

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "transformers==4.48.3"
