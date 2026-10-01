#!/usr/bin/env bash
# Download the ML models and display fonts the pipeline needs (not committed).
# usage: ./fetch_assets.sh [assets_dir]
set -euo pipefail
A="${1:-$(cd "$(dirname "$0")" && pwd)/assets}"
mkdir -p "$A/models" "$A/fonts"

get() {  # url dest
    [ -s "$2" ] || curl -sSLf --retry 3 -o "$2" "$1"
}

get https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Shinkai_37.onnx "$A/models/Shinkai_37.onnx"
get https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx "$A/models/u2net_human_seg.onnx"
get https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task "$A/models/pose_landmarker_heavy.task"

G=https://raw.githubusercontent.com/google/fonts/main/ofl
for f in delagothicone/DelaGothicOne-Regular.ttf bangers/Bangers-Regular.ttf anton/Anton-Regular.ttf \
         shipporiminchob1/ShipporiMinchoB1-ExtraBold.ttf reggaeone/ReggaeOne-Regular.ttf pottaone/PottaOne-Regular.ttf \
         yujiboku/YujiBoku-Regular.ttf mplusrounded1c/MPLUSRounded1c-ExtraBold.ttf; do
    get "$G/$f" "$A/fonts/$(basename "$f")"
done
echo "assets ready in $A"
