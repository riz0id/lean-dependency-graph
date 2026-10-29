#!/bin/bash

input=$(cat)
fp=$(echo "$input" | jq -r '.tool_input.file_path')

if [[ "$fp" == *.py ]]; then
  ruff check --fix "$fp" 2>/dev/null || true
fi

echo "$input"
