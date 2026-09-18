#!/usr/bin/env bash
# Download the model weights Talkover needs into ~/models/.
set -euo pipefail

DEST="${MODELS_DIR:-$HOME/models}"
mkdir -p "$DEST"

hf download openbmb/MiniCPM-o-4_5 --local-dir "$DEST/MiniCPM-o-4_5"
hf download Gander-Omni/Gander --local-dir "$DEST/Gander"
hf download mlx-community/whisper-large-v3-turbo --local-dir "$DEST/whisper-large-v3-turbo"
