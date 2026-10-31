#!/usr/bin/env bash
# Load-test every model with Perf Analyzer (run from the repo root,
# with the Triton server already up: docker compose up -d --build).
#
# The SDK container runs perf_analyzer; on Docker Desktop for Windows it
# reaches the host-published server port via host.docker.internal (-u flag;
# there is no --long form for the URL option in this version).
#
# Notes on synthetic inputs:
#  - embedder: random INT64 ids would violate the Gather bounds, so
#    --input-data zero is used (id 0 = [PAD], valid for load testing);
#  - classifier: full 2-D shape 1,384 must be given explicitly;
#  - generator/pipeline take a real string via --string-data.
set -u

SDK_IMAGE=nvcr.io/nvidia/tritonserver:24.12-py3-sdk
URL=host.docker.internal:8000
mkdir -p results

run() {
  local name="$1"; shift
  echo "=== perf_analyzer: $name ==="
  docker run --rm -v "$(pwd)/results:/results" "$SDK_IMAGE" \
    perf_analyzer -m "$name" -u "$URL" --percentile 95 "$@" \
    2>&1 | tee "results/perf_${name}.txt"
}

# text_embedder: INT64 inputs, synthetic shape [1, 32]
run text_embedder \
  --input-data zero \
  --shape input_ids:1,32 \
  --shape attention_mask:1,32 \
  --shape token_type_ids:1,32 \
  --concurrency-range 1:8:2

# text_classifier: FP32 input, full 2-D shape [1, 384]
run text_classifier \
  --shape embedding:1,384 \
  --concurrency-range 1:8:2

# text_generator: one short string (CPU LLM is slow -> long windows)
run text_generator \
  --string-data "What is the capital of France?" \
  --concurrency-range 1:2:1 \
  --measurement-interval 30000

# pipeline: the full BLS chain
run pipeline \
  --string-data "What is the capital of France?" \
  --concurrency-range 1:2:1 \
  --measurement-interval 30000

echo "All logs saved to ./results/"
