"""Per-frame preprocessing: anime stylization + U2Net person masks for every source frame.

usage: python3 preprocess.py <work_dir>     (expects <work_dir>/full/f001.png ...)
"""
import os, sys, time
import numpy as np, cv2
from stylize import anime, mask

S = sys.argv[1]
os.makedirs(f'{S}/anime', exist_ok=True)
os.makedirs(f'{S}/masks', exist_ok=True)
n = len([f for f in os.listdir(f'{S}/full') if f.endswith('.png')])
t0 = time.time()
for i in range(1, n + 1):
    img = cv2.imread(f'{S}/full/f{i:03d}.png')
    if not os.path.exists(f'{S}/anime/f{i:03d}.png'):
        cv2.imwrite(f'{S}/anime/f{i:03d}.png', anime(img))
    if not os.path.exists(f'{S}/masks/f{i:03d}.png'):
        cv2.imwrite(f'{S}/masks/f{i:03d}.png', (mask(img) * 255).astype(np.uint8))
    print(f'frame {i}/{n}  {time.time() - t0:.0f}s', flush=True)
