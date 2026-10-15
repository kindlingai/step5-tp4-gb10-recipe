#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REV=bc246cc2848e17b37f377eda8fd825c5b3018d1f
mkdir -p "$HERE/tokenizer"
for file in tokenizer.json tokenizer_config.json chat_template.jinja; do
  curl -fL --retry 3 "https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4/resolve/$REV/$file" -o "$HERE/tokenizer/$file"
done
