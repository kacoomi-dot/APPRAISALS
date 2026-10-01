# 第十話「十月の門を蹴破れ！」: YT kicks down the Gates of October Sales

A 21.7 s vertical (1080×1920, 30 fps) anime-style short built from a 3.6 s phone clip of Yt's
ready stance, front kick and walk-off. Everything is generated locally: the anime look comes from AnimeGANv3,
and the gate art, effects, lettering and soundtrack are procedural Python.

**The source clip and the rendered video are not committed.** This repository is public and the footage
shows a real person, so both stay local (see `.gitignore`). Rebuild with the original clip:

```bash
sudo apt-get install -y ffmpeg libegl1 libgles2          # libEGL is required by MediaPipe
pip install numpy opencv-python-headless pillow onnxruntime mediapipe scipy
./make_video.sh path/to/20261001_114406.mp4               # -> work/yt_october_sales_anime.mp4
```

`make_video.sh` fetches models and fonts into `assets/` (`fetch_assets.sh`), then runs the steps below.
A full build takes about 10 minutes on 4 CPU cores.

## Pipeline

| Step | Script | What it does |
| --- | --- | --- |
| Frames | `ffmpeg` | 102 auto-rotated 1080×1920 frames |
| Anime style + masks | `src/preprocess.py`, `src/stylize.py` | Edge-preserving smoothing, then AnimeGANv3 *Shinkai* at 720×1280. U2Net human segmentation for the cut-outs |
| Mask cleanup | `src/refine_masks.py` | U2Net merges the black cabinet behind Yt with his trousers (frames 1–50). The mask is intersected with MediaPipe's pose segmentation, and head/body positions go to `track.json` |
| Close-up | `src/closeup.py` | Separate high-res anime pass on the face crop for the glasses-glint shot |
| Gate art | `src/gate.py` | Procedural stone gate, chained doors, padlock, sunset sky, castle walls |
| Picture | `src/render.py`, `src/fx.py` | Ten scenes composited per frame: speed lines, auras, impact frames, lettering, particles |
| Sound | `src/audio.py` | Synthesized score and SFX (taiko, whooshes, gate impact, theme on the IV–V–iii–vi progression) |
| Encode | `ffmpeg` | H.264 CRF 19 + AAC 192 kbps |

`src/timeline.py` holds the scene list that picture and sound share:

| # | Scene | Beat |
| --- | --- | --- |
| 1 | `cold_open` | "Legend says the Gates of October Sales… can only be opened by force." |
| 2 | `ep_title` | 第十話「十月の門を蹴破れ！」, *Episode 10: Kick Down the Gates of October* |
| 3 | `yt_intro` | Ready stance with a blue aura |
| 4 | `char_card` | Freeze-frame character card: YT 参上!! ("YT has arrived!!") |
| 5 | `power_up` | ゴゴゴゴ menace, then the glasses glint: "October… you're MINE." |
| 6 | `charge` | 必殺 十月キック!! ("Special move: OCTOBER KICK!!") |
| 7 | `gate_hit` | The gate shudders, impact frames, the doors blow off toward the camera: ドゴォォン!! |
| 8 | `title_slam` | OCTOBER SALES 開門!! ("The gates are open!!") |
| 9 | `walk_off` | "Now let's go sell some cars." |
| 10 | `tsuzuku` | つづく / To be continued… |

Frame numbers and screen coordinates (glasses, crop origin) are tuned to this specific clip.

## Credits and licenses

- [AnimeGANv3](https://github.com/TachibanaYoshino/AnimeGANv3) `Shinkai_37` weights: free for non-commercial use.
  Commercial use (for example, dealership advertising) needs the author's written permission.
- U2Net human segmentation weights from the [rembg](https://github.com/danielgatis/rembg) releases.
- [MediaPipe](https://ai.google.dev/edge/mediapipe) Pose Landmarker (heavy).
- Fonts (SIL Open Font License, via Google Fonts): Dela Gothic One, Anton, Bangers, Shippori Mincho B1,
  Yuji Boku, M PLUS Rounded 1c, Reggae One, Potta One.
