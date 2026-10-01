#!/usr/bin/env bash
# Rebuild the anime short from the original 3.6 s phone clip.
# usage: ./make_video.sh path/to/source.mp4 [work_dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(realpath "$1")"
WORK="$(realpath -m "${2:-$HERE/work}")"
WORKERS="${WORKERS:-4}"

"$HERE/fetch_assets.sh" "$HERE/assets"
mkdir -p "$WORK/full"
cp "$HERE"/src/*.py "$WORK/"
ln -sfn "$HERE/assets/fonts" "$WORK/fonts"
ln -sfn "$HERE/assets/models" "$WORK/models"
cd "$WORK"

[ -f full/f001.png ] || ffmpeg -v error -y -i "$SRC" -vsync 0 full/f%03d.png   # 102 frames, auto-rotated to 1080x1920
python3 preprocess.py "$WORK"                     # AnimeGANv3 Shinkai style + U2Net person masks
python3 refine_masks.py "$WORK"                   # MediaPipe pose mask drops the cabinet; writes track.json
python3 closeup.py "$WORK"                        # high-res anime pass for the glasses close-up
python3 gate.py "$WORK/gate_assets.npz" "$WORK/gate_preview.jpg"

TOTAL=$(python3 -c 'from timeline import TOTAL; print(TOTAL)')
for ((w = 0; w < WORKERS; w++)); do
    OMP_NUM_THREADS=1 python3 render.py range $((w * TOTAL / WORKERS)) $(((w + 1) * TOTAL / WORKERS)) > "render_$w.log" &
done
wait
python3 audio.py soundtrack.wav

ffmpeg -v error -y -framerate 30 -i out_frames/f%04d.png -i soundtrack.wav -map 0:v -map 1:a \
    -c:v libx264 -preset slow -crf 19 -profile:v high -pix_fmt yuv420p -tune animation -movflags +faststart \
    -c:a aac -b:a 192k -ar 48000 -shortest -metadata title="YT Kicks Down the Gates of October Sales" \
    yt_october_sales_anime.mp4
echo "done: $WORK/yt_october_sales_anime.mp4"
