#!/usr/bin/env bash
set -e

REPO_ID="System2-Online/Qwen3.8-27B-DualProcess-GGUF"
LOG_FILE="/tmp/hf_upload.log"

echo "=== System2 Model Upload Initiated: $(date) ===" | tee "$LOG_FILE"

MODELS=(
  "/Users/thomjames-worrall/.lmstudio/models/unsloth/Qwen-3.8-27B-Intermediate/Qwen3.8-27B-UD-IQ3_XXS.gguf:Qwen3.8-27B-UD-IQ3_XXS.gguf"
  "/Users/thomjames-worrall/.lmstudio/models/unsloth/Qwen-3.8-27B-Intermediate/Qwen3.8-27B-UD-Q2_K_XL.gguf:Qwen3.8-27B-UD-Q2_K_XL.gguf"
  "/Users/thomjames-worrall/.lmstudio/models/unsloth/Qwen-3.8-27B-Smoll/Qwen3.8-27B-UD-IQ2_XXS.gguf:Qwen3.8-27B-UD-IQ2_XXS.gguf"
)

for item in "${MODELS[@]}"; do
  LOCAL_PATH="${item%%:*}"
  REMOTE_PATH="${item##*:}"
  
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Starting upload: $REMOTE_PATH ($(ls -lh "$LOCAL_PATH" | awk '{print $5}'))" | tee -a "$LOG_FILE"
  echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
  
  hf upload "$REPO_ID" "$LOCAL_PATH" "$REMOTE_PATH" \
    --commit-message "Upload $REMOTE_PATH" 2>&1 | tee -a "$LOG_FILE"
    
  echo ">>> Finished upload: $REMOTE_PATH at $(date)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "=== All System2 Models Uploaded Successfully at $(date) ===" | tee -a "$LOG_FILE"
