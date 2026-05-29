#!/usr/bin/env bash
# Compile ONNX artifact to TensorRT engine (requires trtexec on PATH).
set -euo pipefail

ONNX_PATH="${1:-artifacts/hybrid.onnx}"
ENGINE_PATH="${2:-artifacts/hybrid.engine}"

trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --minShapes=time_series:1x30x6,text_per_step:1x30x32 \
  --optShapes=time_series:8x30x6,text_per_step:8x30x32 \
  --maxShapes=time_series:32x30x6,text_per_step:32x30x32

echo "TensorRT engine saved to ${ENGINE_PATH}"
