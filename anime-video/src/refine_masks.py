# Remove the black cabinet that U2Net merges with Yt's dark trousers (frames 1-50):
# intersect U2Net's mask with a dilated MediaPipe pose-segmentation mask.
import mediapipe as mp, numpy as np, cv2, sys, os, shutil, json
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
S = sys.argv[1]
os.makedirs(f'{S}/masks_u2net', exist_ok=True)
opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=f'{S}/models/pose_landmarker_heavy.task'),
                                    running_mode=vision.RunningMode.IMAGE, output_segmentation_masks=True, num_poses=1)
lm = vision.PoseLandmarker.create_from_options(opts)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
for i in range(1, 51):
    src = f'{S}/masks/f{i:03d}.png'; bak = f'{S}/masks_u2net/f{i:03d}.png'
    if not os.path.exists(bak):
        shutil.copy(src, bak)
    um = cv2.imread(bak, 0).astype(np.float32) / 255
    bgr = cv2.imread(f'{S}/full/f{i:03d}.png')
    r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    if not r.segmentation_masks:
        print(i, 'no pose; keeping u2net'); continue
    pm = r.segmentation_masks[0].numpy_view().astype(np.float32)
    if pm.ndim == 3: pm = pm[..., 0]
    gate_ = cv2.dilate((pm > 0.5).astype(np.uint8), k).astype(np.float32)
    gate_ = cv2.GaussianBlur(gate_, (0, 0), 4)
    out = um * gate_
    # keep only the largest connected component (drop specks)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((out > 0.5).astype(np.uint8))
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep = cv2.dilate((lab == big).astype(np.uint8), k).astype(np.float32)
        out = out * cv2.GaussianBlur(keep, (0, 0), 3)
    cv2.imwrite(src, (np.clip(out, 0, 1) * 255).astype(np.uint8))
    print(i, 'removed px:', int(((um > 0.5) & (out <= 0.5)).sum()), flush=True)
lm.close()
# recompute tracking
info = {}
for i in range(1, 103):
    m = cv2.imread(f'{S}/masks/f{i:03d}.png', 0).astype(np.float32) / 255
    ys, xs = np.where(m > 0.5)
    top, bot = ys.min(), ys.max(); left, right = xs.min(), xs.max()
    hh = int((bot - top) * 0.11)
    by, bx = np.where(m[top:top + hh] > 0.5)
    info[i] = dict(top=int(top), bot=int(bot), left=int(left), right=int(right),
                   head=(float(bx.mean()), float(top + hh * 0.55)), center=(float(xs.mean()), float(ys.mean())))
json.dump(info, open(f'{S}/track.json', 'w'))
print('done')
