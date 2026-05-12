#!/bin/sh
set -e

OPENVINO_LIBS=$(cat /openvino_libs_path)
export LD_LIBRARY_PATH="${OPENVINO_LIBS}:${LD_LIBRARY_PATH}"

echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

exec uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1 --timeout-keep-alive 120