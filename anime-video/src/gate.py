"""Procedural anime-style art for 'The Gates of October Sales'."""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw
import fx
from fx import W, H, hexc

SS = 2  # supersampling
OUT = '#1a0f14'


def C(h, a=255):
    return tuple(int(c * 255) for c in hexc(h)) + (a,)


def _down(img):
    return fx.rgba_from_pil(img.resize((W, H), Image.LANCZOS))


def _s(v):
    return v * SS


# door opening geometry (1x coords)
DX0, DX1, SEAM = 140, 940, 540
DY0, DY1 = 430, 1800
EMB = (540, 800, 205)          # emblem centre + radius
LOCK = (540, 1175)


def frame_layer():
    img = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    rng = np.random.default_rng(7)
    # pillars
    for x0, x1 in ((0, DX0), (DX1, W)):
        d.rectangle([_s(x0), _s(120), _s(x1), _s(H)], fill=C('#6f6a80'))
        y = 120; row = 0
        while y < H:
            bh = 150
            off = 0 if row % 2 == 0 else -60
            xs = list(range(x0 + off, x1, 120))
            for bx in xs:
                shade = rng.uniform(-0.06, 0.06)
                base = np.clip(np.array(hexc('#6f6a80')) + shade, 0, 1)
                d.rectangle([_s(max(bx, x0)), _s(y), _s(min(bx + 120, x1)), _s(min(y + bh, H))],
                            fill=tuple(int(c * 255) for c in base) + (255,), outline=C('#3c3850'), width=_s(4))
                # top-left highlight
                d.line([_s(max(bx, x0) + 6), _s(y + 6), _s(min(bx + 120, x1) - 6), _s(y + 6)], fill=C('#9a95b0'), width=_s(3))
            y += bh; row += 1
    # inner shadows of the opening
    d.rectangle([_s(DX0 - 26), _s(DY0), _s(DX0), _s(DY1)], fill=C('#2a2638'))
    d.rectangle([_s(DX1), _s(DY0), _s(DX1 + 26), _s(DY1)], fill=C('#2a2638'))
    # wall above the cornice
    d.rectangle([_s(0), _s(0), _s(W), _s(120)], fill=C('#4b4660'))
    for by in range(-60, 120, 75):
        d.line([_s(0), _s(by), _s(W), _s(by)], fill=C('#3c3850'), width=_s(4))
    # lintel + cornice
    d.rectangle([_s(0), _s(120), _s(W), _s(DY0)], fill=C('#7a7590'), outline=C(OUT), width=_s(6))
    for bx in range(0, W, 180):
        d.line([_s(bx), _s(150), _s(bx), _s(DY0)], fill=C('#4b4660'), width=_s(4))
    d.rectangle([_s(-10), _s(90), _s(W + 10), _s(150)], fill=C('#938eaa'), outline=C(OUT), width=_s(6))
    d.rectangle([_s(-10), _s(DY0 - 34), _s(W + 10), _s(DY0)], fill=C('#5b5672'), outline=C(OUT), width=_s(5))
    # plaque
    px0, py0, px1, py1 = 140, 178, 940, 368
    d.rounded_rectangle([_s(px0), _s(py0), _s(px1), _s(py1)], radius=_s(18), fill=C('#2a1416'), outline=C(OUT), width=_s(8))
    d.rounded_rectangle([_s(px0 + 12), _s(py0 + 12), _s(px1 - 12), _s(py1 - 12)], radius=_s(12), outline=C('#e3b341'), width=_s(7))
    for cx in (px0 + 40, px1 - 40):
        for cy in (py0 + 40, py1 - 40):
            d.ellipse([_s(cx - 11), _s(cy - 11), _s(cx + 11), _s(cy + 11)], fill=C('#e3b341'), outline=C(OUT), width=_s(3))
    # floor
    d.rectangle([_s(0), _s(DY1), _s(W), _s(H)], fill=C('#3d394d'), outline=C(OUT), width=_s(6))
    for i in range(-6, 8):
        d.line([_s(SEAM + i * 40), _s(DY1), _s(SEAM + i * 260), _s(H)], fill=C('#2b2838'), width=_s(4))
    d.line([_s(0), _s(DY1 + 50), _s(W), _s(DY1 + 50)], fill=C('#2b2838'), width=_s(4))
    lay = _down(img)
    # plaque text
    t = fx.text_sprite('OCTOBER SALES', 'anton', 128, grad=('#fff3b0', '#e39b21'),
                       strokes=((10, '#120806'),), tracking=6)
    _blit_rgba(lay, t, (px0 + px1) / 2, (py0 + py1) / 2 + 4)
    return lay


def _blit_rgba(dst_rgba, spr, cx, cy, scale=1.0, angle=0.0):
    """Composite a premultiplied sprite onto a premultiplied RGBA layer."""
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += cx - w / 2; M[1, 2] += cy - h / 2
    wp = cv2.warpAffine(spr, M, (dst_rgba.shape[1], dst_rgba.shape[0]), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = wp[..., 3:4]
    dst_rgba[:] = wp + dst_rgba * (1 - a)


def doors_layer():
    """Both doors + emblem + rings, returned as (left, right) premultiplied RGBA full-frame layers."""
    img = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    rng = np.random.default_rng(11)
    for x0, x1 in ((DX0, SEAM), (SEAM, DX1)):
        d.rectangle([_s(x0), _s(DY0), _s(x1), _s(DY1)], fill=C('#6b2f1c'), outline=C(OUT), width=_s(7))
        pw = (x1 - x0) / 5
        for k in range(5):
            px = x0 + k * pw
            shade = ['#6b2f1c', '#62291a', '#73341f', '#5e2817', '#6f321d'][k]
            d.rectangle([_s(px + 3), _s(DY0 + 4), _s(px + pw - 3), _s(DY1 - 4)], fill=C(shade))
            d.line([_s(px), _s(DY0), _s(px), _s(DY1)], fill=C('#3a150c'), width=_s(4))
            for g in range(3):  # grain
                gx = px + rng.uniform(12, pw - 12)
                pts = [(_s(gx + 6 * math.sin(yy / 90 + g)), _s(yy)) for yy in range(DY0 + 10, DY1 - 10, 30)]
                d.line(pts, fill=C('#4a1d10', 160), width=_s(2))
            d.line([_s(px + 6), _s(DY0 + 6), _s(px + 6), _s(DY1 - 6)], fill=C('#8a4a2c', 170), width=_s(2))
        # iron bands with rivets
        for by in (DY0 + 70, DY0 + 470, DY0 + 900, DY0 + 1240):
            d.rectangle([_s(x0), _s(by), _s(x1), _s(by + 58)], fill=C('#2e2d38'), outline=C(OUT), width=_s(5))
            d.line([_s(x0 + 4), _s(by + 8), _s(x1 - 4), _s(by + 8)], fill=C('#575769'), width=_s(4))
            for rx in np.arange(x0 + 30, x1 - 10, 62):
                d.ellipse([_s(rx - 10), _s(by + 19), _s(rx + 10), _s(by + 39)], fill=C('#8d8ea0'), outline=C(OUT), width=_s(3))
                d.ellipse([_s(rx - 5), _s(by + 22), _s(rx), _s(by + 27)], fill=C('#d8d9e6'))
        # gold trim
        d.rectangle([_s(x0 + 16), _s(DY0 + 16), _s(x1 - 16), _s(DY1 - 16)], outline=C(OUT), width=_s(14))
        d.rectangle([_s(x0 + 18), _s(DY0 + 18), _s(x1 - 18), _s(DY1 - 18)], outline=C('#e3b341'), width=_s(8))
    # emblem
    ex, ey, er = EMB
    d.ellipse([_s(ex - er - 8), _s(ey - er - 8), _s(ex + er + 8), _s(ey + er + 8)], fill=C(OUT))
    d.ellipse([_s(ex - er), _s(ey - er), _s(ex + er), _s(ey + er)], fill=C('#e3b341'))
    d.ellipse([_s(ex - er + 10), _s(ey - er + 10), _s(ex + er - 10), _s(ey + er - 10)], outline=C('#fff0a8'), width=_s(4))
    ri = er - 30
    d.ellipse([_s(ex - ri - 5), _s(ey - ri - 5), _s(ex + ri + 5), _s(ey + ri + 5)], fill=C(OUT))
    d.ellipse([_s(ex - ri), _s(ey - ri), _s(ex + ri), _s(ey + ri)], fill=C('#8e1b1b'))
    r2 = ri - 46
    d.ellipse([_s(ex - r2), _s(ey - r2), _s(ex + r2), _s(ey + r2)], outline=C('#e3b341'), width=_s(5))
    # door rings
    for rx in (SEAM - 52, SEAM + 52):
        ry = 1080 + 0
        d.ellipse([_s(rx - 20), _s(ry - 20), _s(rx + 20), _s(ry + 20)], fill=C('#2e2d38'), outline=C(OUT), width=_s(4))
        d.ellipse([_s(rx - 36), _s(ry - 4), _s(rx + 36), _s(ry + 68)], outline=C(OUT), width=_s(15))
        d.ellipse([_s(rx - 33), _s(ry - 1), _s(rx + 33), _s(ry + 65)], outline=C('#8d8ea0'), width=_s(8))
    lay = _down(img)
    # emblem text: big 10 + ring of small text
    ten = fx.text_sprite('10', 'anton', 250, grad=('#fff3b0', '#d98a16'), strokes=((10, '#120806'),))
    _blit_rgba(lay, ten, ex, ey + 6)
    ring_txt = '★ OCTOBER ★ SALES ★ OCTOBER ★ SALES '
    rr = (ri + r2) / 2 + 1
    for i, ch in enumerate(ring_txt):
        ang = -90 + 360 * i / len(ring_txt)
        a = math.radians(ang)
        spr = fx.text_sprite(ch, 'anton' if ch != '★' else 'dela', 30, fill='#ffd76a', strokes=((3, '#2a0a0a'),), pad=8)
        _blit_rgba(lay, spr, ex + rr * math.cos(a), ey + rr * math.sin(a), angle=-(ang + 90))
    left = lay.copy(); right = lay.copy()
    left[:, SEAM:] = 0; right[:, :SEAM] = 0
    return left, right


def chains_layer():
    img = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    for (x0, y0), (x1, y1) in (((DX0 + 20, 560), (DX1 - 20, 1760)), ((DX1 - 20, 560), (DX0 + 20, 1760))):
        L = math.hypot(x1 - x0, y1 - y0); ux, uy = (x1 - x0) / L, (y1 - y0) / L; px, py = -uy, ux
        n = int(L / 36)
        for i in range(n + 1):
            cx, cy = x0 + ux * i * 36, y0 + uy * i * 36
            if i % 2 == 0:
                pts = [(cx + ux * 26 * math.cos(t) + px * 14 * math.sin(t), cy + uy * 26 * math.cos(t) + py * 14 * math.sin(t))
                       for t in np.linspace(0, 2 * math.pi, 28)]
                d.line([(_s(x), _s(y)) for x, y in pts], fill=C(OUT), width=_s(13), joint='curve')
                d.line([(_s(x), _s(y)) for x, y in pts], fill=C('#9a9bb0'), width=_s(7), joint='curve')
            else:
                a = (cx - ux * 24, cy - uy * 24); b = (cx + ux * 24, cy + uy * 24)
                d.line([(_s(a[0]), _s(a[1])), (_s(b[0]), _s(b[1]))], fill=C(OUT), width=_s(15))
                d.line([(_s(a[0]), _s(a[1])), (_s(b[0]), _s(b[1]))], fill=C('#6d6e82'), width=_s(8))
    return _down(img)


def padlock_sprite():
    S = 300 * SS
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    c = S / 2
    d.arc([c - _s(75), c - _s(140), c + _s(75), c + _s(10)], 180, 360, fill=C(OUT), width=_s(36))
    d.arc([c - _s(70), c - _s(135), c + _s(70), c + _s(5)], 180, 360, fill=C('#9a9bb0'), width=_s(22))
    d.line([c - _s(70), c - _s(65), c - _s(70), c - _s(10)], fill=C('#9a9bb0'), width=_s(22))
    d.line([c + _s(70), c - _s(65), c + _s(70), c - _s(10)], fill=C('#9a9bb0'), width=_s(22))
    d.rounded_rectangle([c - _s(115), c - _s(30), c + _s(115), c + _s(140)], radius=_s(28), fill=C('#e3b341'), outline=C(OUT), width=_s(9))
    d.rounded_rectangle([c - _s(98), c - _s(14), c + _s(98), c + _s(124)], radius=_s(20), outline=C('#fff0a8'), width=_s(4))
    d.ellipse([c - _s(22), c + _s(20), c + _s(22), c + _s(64)], fill=C(OUT))
    d.polygon([(c - _s(12), c + _s(50)), (c + _s(12), c + _s(50)), (c + _s(18), c + _s(100)), (c - _s(18), c + _s(100))], fill=C(OUT))
    img = img.resize((300, 300), Image.LANCZOS)
    return fx.rgba_from_pil(img)


def seam_glow(intensity=1.0, col='#ffb347'):
    xx, yy = fx.grid()
    dx = np.abs(xx - SEAM)
    core = np.exp(-(dx / 5.0) ** 2)
    halo = np.exp(-(dx / 60.0) ** 2) * 0.55
    vmask = fx.smoothstep(DY0 - 20, DY0 + 40, yy) * (1 - fx.smoothstep(DY1 - 40, DY1 + 20, yy))
    c = np.array(hexc(col), np.float32)
    lay = (core[..., None] * np.array([1, 0.97, 0.9], np.float32) + halo[..., None] * c) * vmask[..., None]
    # light also leaks around the door edges
    edge = (np.exp(-((xx - DX0) / 10) ** 2) + np.exp(-((xx - DX1) / 10) ** 2)) * 0.35 * vmask
    edge += np.exp(-((yy - DY1) / 10) ** 2) * 0.5 * ((xx > DX0) & (xx < DX1))
    lay += edge[..., None] * c
    return np.clip(lay * intensity, 0, 1).astype(np.float32)


def sky():
    xx, yy = fx.grid()
    stops = [(0, '#070b24'), (520, '#26164a'), (900, '#6a2457'), (1200, '#c2410c'), (1420, '#ff9a3c'), (1600, '#ffd27a')]
    ys = np.arange(H, dtype=np.float32)
    col = np.zeros((H, 3), np.float32)
    for (y0, c0), (y1, c1) in zip(stops[:-1], stops[1:]):
        t = np.clip((ys - y0) / (y1 - y0), 0, 1)[:, None]
        seg = (ys >= y0) & (ys < y1)
        col[seg] = (np.array(hexc(c0)) * (1 - t) + np.array(hexc(c1)) * t)[seg]
    col[ys >= stops[-1][0]] = hexc(stops[-1][1])
    img = np.repeat(col[:, None, :], W, axis=1).astype(np.float32)
    # big sun
    r = np.sqrt((xx - 540) ** 2 + (yy - 1180) ** 2)
    sun = 1 - fx.smoothstep(465, 475, r)
    img = img * (1 - sun[..., None]) + np.array(hexc('#ffe39a'), np.float32) * sun[..., None]
    glow = np.exp(-np.maximum(r - 470, 0) / 260)[..., None] * np.array(hexc('#ff8a3c'), np.float32) * 0.55
    img = 1 - (1 - img) * (1 - glow)
    # stars
    rng = np.random.default_rng(5)
    for _ in range(140):
        sx, sy = rng.integers(0, W), rng.integers(0, 760)
        b = rng.uniform(0.3, 1.0) * (1 - sy / 800)
        cv2.circle(img, (int(sx), int(sy)), int(rng.integers(1, 3)), (b, b, b * 0.95), -1, cv2.LINE_AA)
    # streaky anime clouds
    cl = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(cl)
    for (cx, cy, w, h, top, bot) in [(180, 980, 520, 60, '#4b2463', '#ff8a5c'), (880, 900, 600, 70, '#4b2463', '#ff9a6a'),
                                      (600, 1080, 760, 46, '#6a2457', '#ffb27a'), (60, 1180, 420, 40, '#7a2a50', '#ffc08a'),
                                      (980, 1150, 380, 38, '#7a2a50', '#ffc08a'), (420, 760, 420, 40, '#2e1a52', '#c25a6a')]:
        for k in range(5):
            ox = (k - 2) * w * 0.18; oy = -abs(k - 2) * h * 0.18
            d.ellipse([_s(cx + ox - w * 0.32), _s(cy + oy - h), _s(cx + ox + w * 0.32), _s(cy + oy + h)], fill=C(top))
        d.ellipse([_s(cx - w / 2), _s(cy), _s(cx + w / 2), _s(cy + h * 0.9)], fill=C(bot))
    cl = fx.rgba_from_pil(cl.resize((W, H), Image.LANCZOS))
    img = img * (1 - cl[..., 3:4]) + cl[..., :3]
    return np.clip(img, 0, 1).astype(np.float32)


def walls():
    """Silhouetted castle walls flanking the gate (for the wide establishing shot)."""
    img = Image.new('RGBA', (W * SS, H * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    top = 1420
    d.rectangle([_s(-10), _s(top), _s(W + 10), _s(H)], fill=C('#1c1530'))
    for x in range(-10, W, 70):
        d.rectangle([_s(x), _s(top - 46), _s(x + 40), _s(top)], fill=C('#1c1530'))
    d.line([_s(0), _s(top + 6), _s(W), _s(top + 6)], fill=C('#ff9a5c', 120), width=_s(3))
    # ground path
    d.polygon([(_s(330), _s(1690)), (_s(750), _s(1690)), (_s(1300), _s(H)), (_s(-220), _s(H))], fill=C('#2a2240'))
    d.rectangle([_s(-10), _s(1690), _s(W + 10), _s(1700)], fill=C('#140f22'))
    return fx.rgba_from_pil(img.resize((W, H), Image.LANCZOS))


def build_all():
    left, right = doors_layer()
    return dict(frame=frame_layer(), door_l=left, door_r=right, chains=chains_layer(),
                lock=padlock_sprite(), sky=sky(), walls=walls())


if __name__ == '__main__':
    import sys
    A = build_all()
    np.savez_compressed(sys.argv[1], **A)
    # preview
    img = A['sky'].copy()
    g = np.zeros((H, W, 4), np.float32)
    for k in ('frame', 'door_l', 'door_r', 'chains'):
        g = A[k] + g * (1 - A[k][..., 3:4])
    _blit_rgba(g, A['lock'], LOCK[0], LOCK[1])
    close = np.zeros((H, W, 3), np.float32) + 0.2
    close = close * (1 - g[..., 3:4]) + g[..., :3]
    fx.screen(close, seam_glow(0.9))
    wide = A['sky'].copy()
    wl = A['walls']; wide = wide * (1 - wl[..., 3:4]) + wl[..., :3]
    fx.blit(wide, g, 540, 1690 - H * 0.72 / 2, scale=0.72)
    cv2.imwrite(sys.argv[2], cv2.cvtColor((np.hstack([close, wide]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)[::2, ::2])
