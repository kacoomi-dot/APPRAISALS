"""Anime FX toolkit: compositing, text, speed lines, sunbursts, auras, particles, gate art.

All layers are float32 in [0, 1]. RGB frames are (H, W, 3). Sprites are premultiplied
RGBA (h, w, 4). Frame size is portrait 1080x1920.
"""
import math, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, 'fonts')
FONTS = {
    'dela':    f'{FONT_DIR}/DelaGothicOne-Regular.ttf',     # heavy JP gothic (SFX)
    'bangers': f'{FONT_DIR}/Bangers-Regular.ttf',           # comic latin
    'anton':   f'{FONT_DIR}/Anton-Regular.ttf',             # condensed latin titles
    'mincho':  f'{FONT_DIR}/ShipporiMinchoB1-ExtraBold.ttf', # episode-title mincho
    'reggae':  f'{FONT_DIR}/ReggaeOne-Regular.ttf',
    'potta':   f'{FONT_DIR}/PottaOne-Regular.ttf',           # brushy JP
    'brush':   f'{FONT_DIR}/YujiBoku-Regular.ttf',           # calligraphy JP
    'round':   f'{FONT_DIR}/MPLUSRounded1c-ExtraBold.ttf',  # fansub
}
_font_cache = {}


def font(name, size):
    key = (name, int(size))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(FONTS[name], int(size))
    return _font_cache[key]


def hexc(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ease_out_back(t, s=1.70158):
    t = min(max(t, 0.0), 1.0) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_in_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return t ** 3


def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- compositing

def blit(dst, spr, cx, cy, scale=1.0, angle=0.0, alpha=1.0, mode='over'):
    """Composite premultiplied RGBA sprite onto dst, centred at (cx, cy)."""
    if spr is None or scale <= 1e-3 or alpha <= 1e-3:
        return
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    x0 = int(max(0, math.floor(corners[:, 0].min()))); x1 = int(min(dst.shape[1], math.ceil(corners[:, 0].max())))
    y0 = int(max(0, math.floor(corners[:, 1].min()))); y1 = int(min(dst.shape[0], math.ceil(corners[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
    wp = cv2.warpAffine(spr, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    roi = dst[y0:y1, x0:x1]
    if mode == 'over':
        a = wp[..., 3:4] * alpha
        roi[:] = roi * (1 - a) + wp[..., :3] * alpha
    elif mode == 'add':
        roi[:] = np.minimum(1, roi + wp[..., :3] * alpha)
    elif mode == 'screen':
        roi[:] = 1 - (1 - roi) * (1 - wp[..., :3] * alpha)


def over_mask(dst, color, a):
    """dst = dst*(1-a) + color*a ; a is (H,W) float, color is rgb tuple or (H,W,3)."""
    a3 = a[..., None]
    dst *= (1 - a3)
    dst += np.asarray(color, np.float32) * a3 if not isinstance(color, np.ndarray) or color.ndim == 1 else color * a3


def screen(dst, layer, k=1.0):
    dst[:] = 1 - (1 - dst) * (1 - np.clip(layer * k, 0, 1))


def add(dst, layer, k=1.0):
    np.minimum(1, dst + layer * k, out=dst)


def premul(rgb, a):
    return np.dstack([rgb * a[..., None], a]).astype(np.float32)


def rgba_from_pil(img):
    arr = np.asarray(img.convert('RGBA'), np.float32) / 255
    arr[..., :3] *= arr[..., 3:4]
    return arr


def cam(frame, zoom=1.0, cx=W / 2, cy=H / 2, dx=0.0, dy=0.0, angle=0.0):
    """Virtual camera: zoom about (cx, cy), then translate / rotate. Output same size."""
    if abs(zoom - 1) < 1e-4 and abs(dx) < 1e-3 and abs(dy) < 1e-3 and abs(angle) < 1e-4:
        return frame
    M = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    M[0, 2] += (W / 2 - cx) + dx
    M[1, 2] += (H / 2 - cy) + dy
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def bloom(img, thresh=0.72, sigma=28, k=0.55):
    lum = img.max(axis=2)
    br = img * smoothstep(thresh, 1.0, lum)[..., None]
    small = cv2.resize(br, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
    glow = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    screen(img, glow, k)
    return img


_vig = None


def vignette(img, k=0.35):
    global _vig
    if _vig is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
        _vig = np.clip(1 - smoothstep(0.55, 1.25, r), 0, 1).astype(np.float32)
    img *= (1 - k + k * _vig)[..., None]
    return img


def grade(img, sat=1.25, contrast=1.08, warm=0.0, lift=0.0):
    lum = (img @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    img = lum + (img - lum) * sat
    img = (img - 0.5) * contrast + 0.5 + lift
    if warm:
        img = img * np.array([1 + warm, 1 + warm * 0.35, 1 - warm * 0.6], np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def chroma_shift(img, px=6):
    if px < 1:
        return img
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], int(px), axis=1)
    out[..., 2] = np.roll(img[..., 2], -int(px), axis=1)
    return out


# ----------------------------------------------------------------------------- text

def text_sprite(text, fname, size, fill='#ffffff', strokes=((14, '#000000'),), grad=None, pad=None,
                shadow=None, tracking=0):
    """Render text with layered strokes (outermost first). grad = (top_hex, bottom_hex)."""
    f = font(fname, size)
    maxw = max([s[0] for s in strokes] + [0])
    pad = pad if pad is not None else maxw + 24
    probe = ImageDraw.Draw(Image.new('L', (8, 8)))
    if tracking:
        chars = list(text)
        widths = [probe.textlength(c, font=f) for c in chars]
        total = sum(widths) + tracking * (len(chars) - 1)
        bb = probe.textbbox((0, 0), text, font=f)
        tw, th_top, th_bot = total, bb[1], bb[3]
    else:
        bb = probe.textbbox((0, 0), text, font=f)
        tw, th_top, th_bot = bb[2] - bb[0], bb[1], bb[3]
    w = int(tw + 2 * pad + 4); h = int(th_bot - th_top + 2 * pad + 4)
    ox = pad - (0 if tracking else bb[0]); oy = pad - th_top

    def draw_mask(sw):
        m = Image.new('L', (w, h), 0); d = ImageDraw.Draw(m)
        if tracking:
            x = ox
            for c, cw in zip(chars, widths):
                d.text((x, oy), c, font=f, fill=255, stroke_width=sw, stroke_fill=255)
                x += cw + tracking
        else:
            d.text((ox, oy), text, font=f, fill=255, stroke_width=sw, stroke_fill=255)
        return np.asarray(m, np.float32) / 255

    out = np.zeros((h, w, 4), np.float32)

    def comp(col, a):
        c = np.asarray(col, np.float32) if isinstance(col, np.ndarray) else np.array(hexc(col), np.float32)
        out[..., :3] = c * a[..., None] + out[..., :3] * (1 - a[..., None])
        out[..., 3] = a + out[..., 3] * (1 - a)

    if shadow:
        dx, dy, col = shadow
        a = draw_mask(maxw)
        a = np.roll(np.roll(a, dy, 0), dx, 1)
        comp(col, a)
    for sw, col in strokes:
        comp(col, draw_mask(sw))
    a = draw_mask(0)
    if grad:
        top, bot = np.array(hexc(grad[0])), np.array(hexc(grad[1]))
        ys = np.linspace(0, 1, h)[:, None]
        g0 = np.clip((ys * h - oy) / max(1, th_bot - th_top), 0, 1)
        col = (top[None, None, :] * (1 - g0[..., None]) + bot[None, None, :] * g0[..., None]) * np.ones((h, w, 1))
        comp(col.astype(np.float32), a)
    else:
        comp(fill, a)
    return out


def vtext_sprite(text, fname, size, fill='#ffffff', strokes=((14, '#000000'),), spacing=0.92, grad=None):
    """Vertical (tategaki) text, one glyph per row; rotates long-vowel marks."""
    rows = []
    for ch in text:
        if ch in 'ー〜…':
            s = text_sprite(ch, fname, size, fill, strokes, grad=grad)
            s = np.ascontiguousarray(np.rot90(s, -1))
        else:
            s = text_sprite(ch, fname, size, fill, strokes, grad=grad)
        rows.append(s)
    cw = max(r.shape[1] for r in rows)
    step = int(size * spacing)
    pad = rows[0].shape[0]
    hgt = step * (len(rows) - 1) + pad
    out = np.zeros((hgt, cw, 4), np.float32)
    for i, r in enumerate(rows):
        y0 = i * step + (pad - r.shape[0]) // 2
        x0 = (cw - r.shape[1]) // 2
        y1 = min(hgt, y0 + r.shape[0])
        reg = out[y0:y1, x0:x0 + r.shape[1]]
        src = r[:y1 - y0]
        reg[:] = src + reg * (1 - src[..., 3:4])
    return out


# ----------------------------------------------------------------------------- line FX

def focus_lines(cx, cy, seed, n=150, r_in=(330, 560), wid=(3, 15), ell=(1.0, 1.35), w=W, h=H, ss=2):
    """Manga 'shuuchuu-sen' concentration lines. Returns alpha (h, w)."""
    rng = np.random.default_rng(seed)
    R = math.hypot(max(cx, w - cx), max(cy, h - cy)) + 80
    img = Image.new('L', (w * ss, h * ss), 0); d = ImageDraw.Draw(img)
    for _ in range(n):
        th = rng.uniform(0, 2 * math.pi)
        r0 = rng.uniform(*r_in)
        wb = rng.uniform(*wid)
        dth = wb / (2 * R)
        tip = (cx + r0 * math.cos(th) * ell[0], cy + r0 * math.sin(th) * ell[1])
        p1 = (cx + R * math.cos(th - dth), cy + R * math.sin(th - dth))
        p2 = (cx + R * math.cos(th + dth), cy + R * math.sin(th + dth))
        d.polygon([(x * ss, y * ss) for x, y in (tip, p1, p2)], fill=255)
    a = np.asarray(img, np.float32) / 255
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


def streaks(seed, angle_deg=-75, n=90, length=(300, 1100), wid=(2, 9), w=W, h=H, ss=2):
    """Parallel motion speed-lines across the frame. Returns alpha."""
    rng = np.random.default_rng(seed)
    img = Image.new('L', (w * ss, h * ss), 0); d = ImageDraw.Draw(img)
    th = math.radians(angle_deg); ux, uy = math.cos(th), math.sin(th); px, py = -uy, ux
    diag = math.hypot(w, h)
    for _ in range(n):
        off = rng.uniform(-diag / 2, diag / 2)
        along = rng.uniform(-diag / 2, diag / 2)
        L = rng.uniform(*length); wb = rng.uniform(*wid)
        cx, cy = w / 2 + px * off + ux * along, h / 2 + py * off + uy * along
        a = (cx - ux * L / 2, cy - uy * L / 2); b = (cx + ux * L / 2, cy + uy * L / 2)
        poly = [(a[0], a[1]), (cx + px * wb / 2, cy + py * wb / 2), (b[0], b[1]), (cx - px * wb / 2, cy - py * wb / 2)]
        d.polygon([(x * ss, y * ss) for x, y in poly], fill=255)
    a = np.asarray(img, np.float32) / 255
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


_grid = None


def grid():
    global _grid
    if _grid is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _grid = (xx, yy)
    return _grid


def sunburst(cx, cy, rot, n=22, c_in='#fff3b0', c_mid='#ffb020', c_out='#e2401c', dark=0.78):
    xx, yy = grid()
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    th = np.arctan2(dy, dx) + rot
    s = np.clip(0.5 + np.sin(th * n) * r / (n * 1.6), 0, 1)
    rr = np.clip(r / 1300, 0, 1)[..., None]
    ci, cm, co = (np.array(hexc(c), np.float32) for c in (c_in, c_mid, c_out))
    base = np.where(rr < 0.35, ci + (cm - ci) * (rr / 0.35), cm + (co - cm) * ((rr - 0.35) / 0.65))
    img = base * (dark + (1 - dark) * s[..., None])
    return img.astype(np.float32)


def god_rays(cx, cy, rot, n=14, spread=1.0, falloff=1400):
    xx, yy = grid()
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    th = np.arctan2(dy, dx)
    v = 0.5 + 0.5 * np.sin(th * n + rot) * np.sin(th * (n * 0.61) - rot * 1.7)
    v = np.clip(v, 0, 1) ** 3
    return (v * np.exp(-r / falloff) * spread).astype(np.float32)


def halftone(shape_h, shape_w, cell=18, angle=0.0, scale_map=None):
    xx, yy = grid()
    xx = xx[:shape_h, :shape_w]; yy = yy[:shape_h, :shape_w]
    c, s = math.cos(angle), math.sin(angle)
    u = (xx * c + yy * s) / cell; v = (-xx * s + yy * c) / cell
    du = u - np.floor(u) - 0.5; dv = v - np.floor(v) - 0.5
    dist = np.sqrt(du * du + dv * dv)
    rad = 0.32 if scale_map is None else 0.48 * scale_map
    return np.clip((rad - dist) * cell * 1.2 + 0.5, 0, 1).astype(np.float32)


def crack_lines(cx, cy, seed, n=14, length=(250, 900), w=W, h=H, width=7):
    rng = np.random.default_rng(seed)
    img = Image.new('L', (w, h), 0); d = ImageDraw.Draw(img)
    def branch(x, y, ang, L, wd, depth):
        pts = [(x, y)]; travelled = 0
        while travelled < L:
            step = rng.uniform(25, 60)
            ang += rng.normal(0, 0.35)
            x += math.cos(ang) * step; y += math.sin(ang) * step
            pts.append((x, y)); travelled += step
            if depth < 2 and rng.random() < 0.12:
                branch(x, y, ang + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9), L * 0.4, max(2, wd * 0.6), depth + 1)
        d.line(pts, fill=255, width=int(max(2, wd)), joint='curve')
    for i in range(n):
        ang = 2 * math.pi * i / n + rng.normal(0, 0.18)
        branch(cx, cy, ang, rng.uniform(*length), width, 0)
    return np.asarray(img, np.float32) / 255


# ----------------------------------------------------------------------------- person FX

_noise_cache = {}


def noise_field(seed, w, h, sigma):
    key = (seed, w, h, sigma)
    if key not in _noise_cache:
        rng = np.random.default_rng(seed)
        n = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigma)
        n = (n - n.min()) / (n.max() - n.min())
        _noise_cache[key] = n
    return _noise_cache[key]


def aura(mask, t, col_in='#ffffff', col_out='#38b6ff', strength=1.0, height=90, seed=3):
    """Flame-like rising aura around a person mask (H, W). Returns additive RGB layer."""
    q = 4
    sw, sh = W // q, H // q
    m = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    k = np.linspace(1.0, 0.15, height // q + 1, dtype=np.float32)[:, None]
    k /= k.sum()
    up = cv2.filter2D(m, -1, k, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    glow = cv2.GaussianBlur(m, (0, 0), 5)
    n1 = noise_field(seed, sw, sh * 2, 3.0)
    n2 = noise_field(seed + 1, sw, sh * 2, 1.4)
    off1 = int(t * 140) % sh; off2 = int(t * 260) % sh
    nz = 0.6 * n1[sh - off1: 2 * sh - off1] + 0.4 * n2[sh - off2: 2 * sh - off2]
    fl = np.clip(up * 2.6 * (nz * 1.6 - 0.25), 0, 1)
    halo = np.clip(glow * 1.7 + fl, 0, 1)
    ci = np.array(hexc(col_in), np.float32); co = np.array(hexc(col_out), np.float32)
    layer = halo[..., None] * (co * 0.85 + ci * 0.15 * glow[..., None])
    layer = cv2.resize(layer, (W, H), interpolation=cv2.INTER_LINEAR)
    layer *= (1 - mask[..., None])                       # glow lives outside the silhouette
    er = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    rim = cv2.GaussianBlur(np.clip(mask - er, 0, 1), (0, 0), 2.0)
    layer += rim[..., None] * ci * 0.55                  # thin back-light rim on the body edge
    return np.clip(layer * strength, 0, 1)


def outline(mask, inner_px=6, outer_px=0):
    """Returns (black_line_alpha, outer_rim_alpha) around a soft mask."""
    m = (mask > 0.5).astype(np.uint8)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner_px + 1, 2 * inner_px + 1))
    d1 = cv2.dilate(m, k1)
    line = np.clip(d1.astype(np.float32) - mask, 0, 1)
    line = cv2.GaussianBlur(line, (0, 0), 0.8)
    rim = None
    if outer_px:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (inner_px + outer_px) + 1,) * 2)
        d2 = cv2.dilate(m, k2)
        rim = cv2.GaussianBlur(np.clip(d2.astype(np.float32) - d1, 0, 1), (0, 0), 0.8)
    return line, rim


def ink_lines(rgb, strength=0.55, sigma=1.1, p=20, eps=0.012, phi=9):
    """XDoG line-art multiplied over an image for a drawn look."""
    g = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    g1 = cv2.GaussianBlur(g, (0, 0), sigma); g2 = cv2.GaussianBlur(g, (0, 0), sigma * 1.6)
    d = (1 + p) * g1 - p * g2
    e = np.where(d >= eps, 1.0, 1 + np.tanh(phi * (d - eps))).astype(np.float32)
    e = np.clip(e, 0, 1)
    return rgb * (1 - strength * (1 - e))[..., None]


def xdog_edges(rgb, sigma=1.3, p=22, eps=0.012, phi=9):
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    g1 = cv2.GaussianBlur(g, (0, 0), sigma); g2 = cv2.GaussianBlur(g, (0, 0), sigma * 1.6)
    d = (1 + p) * g1 - p * g2
    return np.clip(np.where(d >= eps, 1.0, 1 + np.tanh(phi * (d - eps))), 0, 1).astype(np.float32)


def impact_bw(rgb, thresh=0.42, invert=True, tint=None):
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    lum = cv2.GaussianBlur(lum, (0, 0), 1.2)
    b = smoothstep(thresh - 0.04, thresh + 0.04, lum)
    if invert:
        b = 1 - b
    if tint is None:
        return np.repeat(b[..., None], 3, axis=2).astype(np.float32)
    c0, c1 = np.array(hexc(tint[0]), np.float32), np.array(hexc(tint[1]), np.float32)
    return (c0 * (1 - b[..., None]) + c1 * b[..., None]).astype(np.float32)


# ----------------------------------------------------------------------------- sprites

def maple_leaf(size=120, color='#e8551f', edge='#7a1d0e', seed=0):
    ss = 4; S = size * ss
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    cx, cy = S / 2, S * 0.54
    pts = []
    for i in range(720):
        th = 2 * math.pi * i / 720
        lobe = (0.5 + 0.5 * math.cos(5 * (th + math.pi / 2))) ** 3.2
        teeth = (0.5 + 0.5 * math.cos(15 * (th + math.pi / 2))) ** 6
        lower = 0.72 + 0.28 * (-math.sin(th) if math.sin(th) < 0 else 0.3)
        r = (0.30 + 0.62 * lobe * lower + 0.05 * teeth) * S * 0.48
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    d.polygon(pts, fill=tuple(int(c * 255) for c in hexc(color)) + (255,),
              outline=tuple(int(c * 255) for c in hexc(edge)) + (255,), width=3 * ss)
    ec = tuple(int(c * 255) for c in hexc(edge)) + (200,)
    for ang in (-90, -18, -162, 54, 126):
        a = math.radians(ang)
        d.line([(cx, cy), (cx + math.cos(a) * S * 0.36, cy + math.sin(a) * S * 0.36)], fill=ec, width=2 * ss)
    d.line([(cx, cy), (cx + S * 0.03, cy + S * 0.42)], fill=ec, width=3 * ss)
    img = img.resize((size, size), Image.LANCZOS)
    return rgba_from_pil(img)


def sparkle(size=320, core='#ffffff', glow='#bfe8ff'):
    ss = 2; S = size * ss
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    c = S / 2
    def diamond(L, w, ang, col):
        a = math.radians(ang); ux, uy = math.cos(a), math.sin(a); px, py = -uy, ux
        d.polygon([(c + ux * L, c + uy * L), (c + px * w, c + py * w), (c - ux * L, c - uy * L), (c - px * w, c - py * w)], fill=col)
    diamond(S * 0.49, S * 0.035, 0, (255, 255, 255, 255))
    diamond(S * 0.49, S * 0.035, 90, (255, 255, 255, 255))
    diamond(S * 0.22, S * 0.022, 45, (255, 255, 255, 230))
    diamond(S * 0.22, S * 0.022, 135, (255, 255, 255, 230))
    img = img.resize((size, size), Image.LANCZOS)
    arr = rgba_from_pil(img)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    g = np.exp(-(r / 0.22) ** 2) * 0.9
    gc = np.array(hexc(glow), np.float32)
    out = arr.copy()
    out[..., :3] = arr[..., :3] + gc * g[..., None] * (1 - arr[..., 3:4])
    out[..., 3] = np.clip(arr[..., 3] + g * (1 - arr[..., 3]), 0, 1)
    return out


def debris_sprite(seed, size=80, palette=('#5a2a1a', '#7b3b22', '#2b2b33', '#c99a2e')):
    rng = np.random.default_rng(seed)
    ss = 3; S = size * ss
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    n = rng.integers(4, 8)
    angs = np.sort(rng.uniform(0, 2 * math.pi, n))
    pts = [(S / 2 + math.cos(a) * S * rng.uniform(0.2, 0.47), S / 2 + math.sin(a) * S * rng.uniform(0.2, 0.47)) for a in angs]
    col = hexc(palette[rng.integers(0, len(palette))])
    d.polygon(pts, fill=tuple(int(c * 255) for c in col) + (255,), outline=(20, 10, 10, 255), width=3 * ss)
    img = img.resize((size, size), Image.LANCZOS)
    return rgba_from_pil(img)


def soft_blob(size=256, col='#d9c8b0', a=0.6):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    al = np.clip(1 - r, 0, 1) ** 1.6 * a
    return premul(np.ones((size, size, 3), np.float32) * np.array(hexc(col), np.float32), al.astype(np.float32))
