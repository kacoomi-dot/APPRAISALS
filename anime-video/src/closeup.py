"""High-res anime pass on a 540x960 crop around Yt's face in f043, for the 2x glasses-glint close-up.

usage: python3 closeup.py <work_dir>
"""
import sys
import numpy as np, cv2
from stylize import anime, mask

S = sys.argv[1]
X0, Y0 = 296, 80          # crop origin in the 1080x1920 frame; render.py's glare/glint coordinates assume it
img = cv2.imread(f'{S}/full/f043.png')
crop = cv2.resize(img[Y0:Y0 + 960, X0:X0 + 540], (1080, 1920), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f'{S}/closeup43_anime.png', anime(crop))
cv2.imwrite(f'{S}/closeup43_mask.png', (mask(crop) * 255).astype(np.uint8))
