#!/usr/bin/env bash
# Smoke-test the /detect endpoint.
# Usage: ./test_detect.sh [IMAGE_PATH] [PROMPT]
# Example: ./test_detect.sh ~/bookshelf.jpg "book spine"

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
IMAGE="${1:-}"
PROMPT="${2:-book spine}"
THRESHOLD="${THRESHOLD:-0.30}"

# Health check
echo "==> health check"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

if [[ -z "$IMAGE" ]]; then
    echo "No image provided. Pass a path as the first argument to test /detect."
    echo "Example: $0 ~/bookshelf.jpg"
    exit 0
fi

echo "==> detecting '${PROMPT}' in: $IMAGE"
curl -s -X POST "$BASE_URL/detect" \
  -F "image=@${IMAGE};type=image/jpeg" \
  -F "prompt=${PROMPT}" \
  -F "threshold=${THRESHOLD}" \
  -F "max_boxes=60" \
  | python3 -m json.tool
