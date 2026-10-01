"""Compositor for 'YT kicks down the Gates of October Sales' (anime short).

usage:  python3 render.py test <global_frame> [...]      -> preview JPEGs
        python3 render.py range <start> <end>             -> PNG frames into out_frames/
"""
import sys, os, json, math, functools
import numpy as np, cv2
import fx, gate
from fx import W, H, hexc, blit, smoothstep, ease_out_back, ease_out_cubic, ease_in_cubic, ease_in_out
from timeline import FPS, SCENES, START, TOTAL

S = os.path.dirname(os.path.abspath(__file__))
G = dict(np.load(f'{S}/gate_assets.npz'))
TRACK = {int(k): v for k, v in json.load(open(f'{S}/track.json')).items()}
OUT = f'{S}/out_frames'

INK = '#1a0f14'


# ---------------------------------------------------------------------------- sources
@functools.lru_cache(maxsize=16)
def anime(i):
    img = cv2.imread(f'{S}/anime/f{i:03d}.png')[..., ::-1].astype(np.float32) / 255
    return fx.grade(img, sat=1.22, contrast=1.06)


@functools.lru_cache(maxsize=16)
def mask(i):
    m = cv2.imread(f'{S}/masks/f{i:03d}.png', 0).astype(np.float32) / 255
    return smoothstep(0.3, 0.7, m).astype(np.float32)


@functools.lru_cache(maxsize=2)
def closeup():
    img = cv2.imread(f'{S}/closeup43_anime.png')[..., ::-1].astype(np.float32) / 255
    m = cv2.imread(f'{S}/closeup43_mask.png', 0).astype(np.float32) / 255
    return fx.grade(img, sat=1.22, contrast=1.06), smoothstep(0.3, 0.7, m).astype(np.float32)


_spr = {}


def spr(key, fn):
    if key not in _spr:
        _spr[key] = fn()
    return _spr[key]


def fit(sprite, max_w):
    return min(1.0, max_w / sprite.shape[1])


# ---------------------------------------------------------------------------- shared pieces
def sub_sprite(text):
    return spr(('sub', text), lambda: fx.text_sprite(text, 'round', 60, fill='#ffffff',
                                                     strokes=((10, '#000000'),), shadow=(4, 5, '#000000')))


def draw_sub(img, text, k, k0, k1, y=1700):
    if k0 <= k <= k1:
        a = min(1.0, (k - k0 + 1) / 3, (k1 - k + 1) / 3)
        s = sub_sprite(text)
        blit(img, s, W / 2, y, scale=fit(s, 1040), alpha=a)


def leaf_sprites():
    return spr('leaves', lambda: [fx.maple_leaf(120, c, e) for c, e in
                                  [('#e8551f', '#7a1d0e'), ('#f08a1c', '#7a3a0e'), ('#d93020', '#6a120a'), ('#f2b52a', '#8a5a0e')]])


def leaves(img, t, seed, n=20, vy=(150, 280), scale=(0.35, 0.95), wind=-50, alpha=1.0):
    rng = np.random.default_rng(seed)
    sp = leaf_sprites()
    for _ in range(n):
        x0 = rng.uniform(-150, W + 150); y0 = rng.uniform(0, H + 300)
        v = rng.uniform(*vy); s = rng.uniform(*scale)
        amp = rng.uniform(30, 90); w = rng.uniform(1.2, 2.6); ph = rng.uniform(0, 6.28)
        spin = rng.uniform(-170, 170); idx = int(rng.integers(0, len(sp)))
        y = (y0 + v * t) % (H + 300) - 150
        x = (x0 + wind * t + amp * math.sin(w * t + ph) + 200) % (W + 400) - 200
        blit(img, sp[idx], x, y, scale=s * (0.75 + 0.25 * math.cos(w * 1.7 * t + ph)), angle=spin * t + ph * 57, alpha=alpha)


def embers(img, t, seed, n=30, col='#ffb347', speed=(60, 170), size=(2, 5), k=1.0):
    rng = np.random.default_rng(seed)
    lay = np.zeros((H // 2, W // 2, 3), np.float32)
    c = np.array(hexc(col), np.float32)
    for _ in range(n):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        v = rng.uniform(*speed); r = int(rng.integers(*size)); ph = rng.uniform(0, 6.28)
        y = (y0 - v * t) % H; x = x0 + 25 * math.sin(1.4 * t + ph)
        b = 0.55 + 0.45 * math.sin(7 * t + ph)
        cv2.circle(lay, (int(x / 2), int(y / 2)), r, tuple(float(v_) for v_ in c * b), -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.5) * 1.6
    fx.add(img, cv2.resize(lay, (W, H)), k)


def sparkles(img, t, seed, n=10, region=(60, 1020, 60, 900), size=(40, 110)):
    rng = np.random.default_rng(seed)
    sp = spr('sparkle_s', lambda: fx.sparkle(160))
    for _ in range(n):
        x, y = rng.uniform(region[0], region[1]), rng.uniform(region[2], region[3])
        per = rng.uniform(0.6, 1.2); ph = rng.uniform(0, 1)
        u = ((t / per) + ph) % 1.0
        s = math.sin(math.pi * u) ** 2 * rng.uniform(*size) / 160
        blit(img, sp, x, y, scale=s, angle=u * 90, mode='add')


def cutout_layer(i, line_px=6, rim_px=12, rim_col='#ffffff', src=None):
    img, m = src if src is not None else (anime(i), mask(i))
    line, rim = fx.outline(m, line_px, rim_px)
    lay = np.zeros((H, W, 4), np.float32)
    if rim is not None:
        lay[..., :3] = np.array(hexc(rim_col), np.float32) * rim[..., None]
        lay[..., 3] = rim
    lay[..., :3] *= (1 - line[..., None]); lay[..., 3] = line + lay[..., 3] * (1 - line)
    lay[..., :3] = img * m[..., None] + lay[..., :3] * (1 - m[..., None])
    lay[..., 3] = m + lay[..., 3] * (1 - m)
    return lay


def place(bg, lay, scale=1.0, ax=W / 2, ay=H / 2, dx=0.0, dy=0.0, angle=0.0):
    M = cv2.getRotationMatrix2D((ax, ay), angle, scale); M[0, 2] += dx; M[1, 2] += dy
    wp = cv2.warpAffine(lay, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return bg * (1 - wp[..., 3:4]) + wp[..., :3]


def cam(frame, zoom=1.0, cx=W / 2, cy=H / 2, tx=None, ty=None, angle=0.0, dx=0.0, dy=0.0):
    """Zoom/rotate about (cx, cy) and move that point to (tx, ty) (+dx, dy)."""
    tx = cx if tx is None else tx; ty = cy if ty is None else ty
    if abs(zoom - 1) < 1e-4 and abs(angle) < 1e-4 and abs(tx - cx + dx) < 1e-3 and abs(ty - cy + dy) < 1e-3:
        return frame
    M = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    M[0, 2] += tx - cx + dx; M[1, 2] += ty - cy + dy
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def shake(img, k, amp, seed=0, rot=0.0):
    if amp < 0.5:
        return img
    rng = np.random.default_rng(seed * 1000 + k)
    return cam(img, zoom=1 + amp / 900, dx=rng.uniform(-amp, amp), dy=rng.uniform(-amp, amp),
               angle=rng.uniform(-rot, rot))


def flash(img, a, col='#ffffff'):
    if a > 0:
        img[:] = img * (1 - a) + np.array(hexc(col), np.float32) * a
    return img


def focus(key, cx, cy, seed, **kw):
    """Cached concentration-line alpha."""
    k2 = ('focus', key, int(cx) // 8, int(cy) // 8, seed)
    return spr(k2, lambda: fx.focus_lines(cx, cy, seed, **kw))


def stamp_scale(k, k0, dur=5, start=2.6):
    """Slam-in scale curve for text."""
    if k < k0:
        return 0
    u = (k - k0) / dur
    if u >= 1:
        return 1.0
    return start + (1 - start) * ease_out_back(u, 2.2)


def jitter(k, amp, seed):
    rng = np.random.default_rng(seed * 7919 + k)
    return rng.uniform(-amp, amp), rng.uniform(-amp, amp)


# ---------------------------------------------------------------------------- gate composites
def gate_close(glow=1.0, with_doors=True):
    key = ('gate_close', with_doors)
    def build():
        g = np.zeros((H, W, 4), np.float32)
        parts = ('frame', 'door_l', 'door_r', 'chains') if with_doors else ('frame',)
        for p in parts:
            g = G[p] + g * (1 - G[p][..., 3:4])
        if with_doors:
            gate._blit_rgba(g, G['lock'], gate.LOCK[0], gate.LOCK[1])
        return g
    g = spr(key, build)
    return g


def gate_wide_rgba(glow):
    g0 = spr('gw0', lambda: _gate_wide(0.0))
    g1 = spr('gw1', lambda: _gate_wide(1.0))
    return g0 + (g1 - g0) * glow


def _gate_wide(glow):
    g = gate_close().copy()
    rgb = g[..., :3] * 0.78 + np.array([0.05, 0.0, 0.03], np.float32) * g[..., 3:4]   # backlit / darker
    gl = gate.seam_glow(1.0)
    rgb = 1 - (1 - rgb) * (1 - gl * glow * 1.25)
    g[..., :3] = rgb * (g[..., 3:4] > 0)
    return g


def door_sprites():
    def build():
        out = {}
        chains = G['chains']
        for side, key, x0, x1 in (('l', 'door_l', gate.DX0 - 30, gate.SEAM + 2), ('r', 'door_r', gate.SEAM - 2, gate.DX1 + 30)):
            d = G[key].copy()
            ch = chains.copy()
            if side == 'l':
                ch[:, gate.SEAM:] = 0
            else:
                ch[:, :gate.SEAM] = 0
            d = ch + d * (1 - ch[..., 3:4])
            out[side] = np.ascontiguousarray(d[gate.DY0 - 10:gate.DY1 + 10, x0:x1])
            out[side + '_c'] = ((x0 + x1) / 2, (gate.DY0 + gate.DY1) / 2)
        return out
    return spr('doors', build)


def debris_list():
    def build():
        rng = np.random.default_rng(21)
        items = []
        for j in range(46):
            sp = fx.debris_sprite(j, size=int(rng.integers(40, 110)))
            x0 = gate.SEAM + rng.normal(0, 60); y0 = rng.uniform(gate.DY0 + 100, gate.DY1 - 100)
            ang = math.atan2(y0 - 1100, x0 - gate.SEAM + rng.normal(0, 40)) + rng.normal(0, 0.5)
            sp_v = rng.uniform(700, 1900)
            items.append(dict(s=sp, x0=x0, y0=y0, vx=math.cos(ang) * sp_v, vy=math.sin(ang) * sp_v - 300,
                              spin=rng.uniform(-600, 600), z=rng.uniform(0.4, 2.2)))
        return items
    return spr('debris', build)


# ---------------------------------------------------------------------------- scenes
def s_cold_open(k):
    t = k / FPS
    img = G['sky'].copy()
    rays = fx.god_rays(540, 1180, t * 0.3, n=16, spread=0.6, falloff=1600)
    fx.screen(img, rays[..., None] * np.array(hexc('#ffd28a'), np.float32))
    wl = G['walls']
    img = img * (1 - wl[..., 3:4]) + wl[..., :3]
    glow = 0.55 + 0.25 * math.sin(t * 5.0) + 0.15 * math.sin(t * 13.0)
    gsc = 0.66
    blit(img, gate_wide_rgba(glow), 540, 1720 - H * gsc / 2, scale=gsc)
    embers(img, t, seed=9, n=34, col='#ffb060')
    img = fx.bloom(img, 0.7, 30, 0.5)
    img = cam(img, zoom=1.0 + 0.10 * ease_in_out(k / 95), cx=540, cy=1050)
    leaves(img, t, seed=4, n=16, scale=(0.35, 0.8))
    # date (vertical brush text)
    date = spr('date', lambda: fx.vtext_sprite('十月一日', 'brush', 118, fill='#ffffff', strokes=((10, INK),)))
    blit(img, date, 975, 470, alpha=float(np.clip((k - 6) / 8, 0, 1)))
    # narration
    lines = [((12, 50), ('Legend says the Gates', 'of October Sales...')),
             ((53, 95), ('...can only be opened', 'by force.'))]
    for (k0, k1), (l1, l2) in lines:
        if k0 <= k <= k1:
            a = min(1.0, (k - k0 + 1) / 5, (k1 - k + 1) / 5)
            for j, line in enumerate((l1, l2)):
                s = spr(('nar', line), lambda line=line: fx.text_sprite(line, 'mincho', 64, fill='#fff6e6',
                                                                        strokes=((8, '#120a14'),), shadow=(0, 4, '#000000')))
                blit(img, s, W / 2, 1715 + j * 92, alpha=a)
    img = fx.vignette(img, 0.4)
    fade = float(np.clip(k / 10, 0, 1))
    return img * fade


def s_ep_title(k):
    t = k / FPS
    img = np.zeros((H, W, 3), np.float32)
    xx, yy = fx.grid()
    r = np.sqrt((xx - 540) ** 2 + (yy - 900) ** 2)
    img += (np.exp(-r / 900)[..., None] * np.array(hexc('#3a0612'), np.float32))
    L = 110
    items = [
        ('ep', '第十話', 'mincho', 110, 520, '#ffffff'),
        ('t1', '十月の門を', 'mincho', 182, 760, '#ffffff'),
        ('t2', '蹴破れ！', 'mincho', 236, 1020, '#ffffff'),
    ]
    for key, txt, fn, size, y, col in items:
        s = spr(('ept', key), lambda txt=txt, fn=fn, size=size, col=col: fx.text_sprite(txt, fn, size, fill=col, strokes=(), pad=10))
        blit(img, s, L + s.shape[1] / 2, y)
    cv2.line(img, (L, 1230), (W - L, 1230), (1, 1, 1), 3, cv2.LINE_AA)
    e1 = spr('ep_en1', lambda: fx.text_sprite('EPISODE 10', 'anton', 66, fill='#ff4a3a', strokes=(), pad=8, tracking=10))
    e2 = spr('ep_en2', lambda: fx.text_sprite('KICK DOWN THE GATES OF OCTOBER', 'anton', 54, fill='#ffffff', strokes=(), pad=8, tracking=4))
    blit(img, e1, L + e1.shape[1] / 2, 1310)
    sc2 = fit(e2, W - 2 * L)
    blit(img, e2, L + e2.shape[1] * sc2 / 2, 1395, scale=sc2)
    img = cam(img, zoom=1.0 + 0.025 * k / 65, cx=540, cy=900)
    return flash(img, max(0.0, 1 - k / 4))


def s_yt_intro(k):
    t = k / FPS
    src = 1 + 2 * (k // 2)
    img = anime(src).copy(); m = mask(src)
    rays = fx.god_rays(700, 260, t * 0.6, n=12, spread=0.4, falloff=1300)
    fx.screen(img, rays[..., None] * np.array(hexc('#fff1c4'), np.float32))
    fx.add(img, fx.aura(m, t, col_in='#ffffff', col_out='#3fb8ff', strength=0.95))
    embers(img, t, seed=5, n=26, col='#9fe3ff', speed=(120, 260))
    img = fx.bloom(img, 0.75, 26, 0.45)
    hx, hy = TRACK[src]['head']
    img = cam(img, zoom=1.0 + 0.07 * ease_in_out(k / 29), cx=hx, cy=hy + 400)
    return fx.vignette(img, 0.3)


def card_bg():
    def build():
        xx, yy = fx.grid()
        t = np.clip((xx * 0.4 + yy) / (W * 0.4 + H), 0, 1)[..., None]
        img = np.array(hexc('#ffb21a'), np.float32) * (1 - t) + np.array(hexc('#ff3d12'), np.float32) * t
        ht = fx.halftone(H, W, cell=24, angle=math.radians(30), scale_map=np.clip(t[..., 0] * 1.2, 0.2, 1))
        img = img * (1 - 0.25 * ht[..., None])
        d = yy - (H * 0.62 - (xx - W / 2) * 0.35)
        band = smoothstep(-200, -197, d) * (1 - smoothstep(130, 133, d))
        img = img * (1 - band[..., None]) + np.array(hexc('#16121f'), np.float32) * band[..., None]
        st = smoothstep(-236, -233, d) * (1 - smoothstep(-218, -215, d))
        img = img * (1 - st[..., None]) + st[..., None]
        return img.astype(np.float32)
    return spr('card_bg', build)


def s_char_card(k):
    t = k / FPS
    src = 29
    foot = anime(src)
    bg = card_bg()
    # diagonal wipe from footage to card
    xx, yy = fx.grid()
    u = ease_out_cubic(k / 5)
    edge = (xx * 0.6 + yy) - (-300 + u * (W * 0.6 + H + 600))
    wipe = 1 - smoothstep(-4, 4, edge)
    base = foot * (1 - wipe[..., None]) + bg * wipe[..., None]
    lay = spr(('cut', src), lambda: cutout_layer(src, 7, 14))
    sc = 1.0 + 0.045 * ease_out_cubic(k / 68)
    img = place(base, lay, scale=sc, ax=W / 2, ay=H)
    # streak flourish
    if k < 20:
        st = spr(('streaks', k // 2), lambda: fx.streaks(300 + k // 2, angle_deg=-19.3, n=40, length=(500, 1400), wid=(3, 10)))
        a = 0.55 * (1 - k / 20)
        fx.over_mask(img, (1, 1, 1), st * a)
    # YT name
    yt = spr('yt_name', lambda: fx.text_sprite('YT', 'anton', 470, grad=('#ffffff', '#ffe08a'),
                                               strokes=((34, '#16121f'), (18, '#e8202a')), shadow=(14, 16, '#16121f')))
    if k >= 3:
        u = ease_out_back((k - 3) / 9, 1.4)
        x = -420 + (300 + 420) * u
        blit(img, yt, x, 1440, scale=0.9, angle=12)
    sanjo = spr('sanjo', lambda: fx.text_sprite('参上!!', 'dela', 170, grad=('#ff6a4a', '#c40018'),
                                                strokes=((30, '#ffffff'), (16, '#16121f'))))
    sc2 = stamp_scale(k, 15, dur=5, start=2.8)
    if sc2:
        jx, jy = jitter(k, 6 if k < 24 else 0, 3)
        blit(img, sanjo, 560 + jx, 205 + jy, scale=sc2 * fit(sanjo, 900), angle=-5)
    if k >= 24:
        tag = spr('hype', lambda: _tag('HYPE  LV. MAX'))
        a = min(1.0, (k - 24) / 3)
        blit(img, tag, 830, 1590, angle=12, alpha=a, scale=0.9 + 0.1 * ease_out_back(min(1, (k - 24) / 5)))
    draw_sub(img, 'YT has arrived!!', k, 15, 68)
    img = flash(img, max(0.0, 0.85 - k / 4))
    if 15 <= k < 21:
        img = shake(img, k, 14 * (1 - (k - 15) / 6), seed=2)
    return img


def _tag(txt):
    s = fx.text_sprite(txt, 'bangers', 66, fill='#ffffff', strokes=((7, '#16121f'),), pad=26)
    h, w = s.shape[:2]
    box = np.zeros((h, w, 4), np.float32)
    cv2.rectangle(box, (6, 10), (w - 6, h - 10), (*hexc('#16121f'), 1.0), -1)
    cv2.rectangle(box, (12, 16), (w - 12, h - 16), (*hexc('#e8202a'), 1.0), -1)
    return s + box * (1 - s[..., 3:4])


def gogo_sprite():
    return spr('gogo', lambda: fx.vtext_sprite('ゴゴゴゴ', 'dela', 132, fill='#7a3cff',
                                               strokes=((22, '#ffffff'), (12, '#16121f')), spacing=0.95))


def s_power_up(k):
    t = k / FPS
    if k < 36:
        src = 29 + k // 2
        img = anime(src).copy(); m = mask(src)
        dark = 0.38 * min(1.0, k / 10)
        img = img * (1 - dark * (1 - m[..., None])) + np.array(hexc('#1b0b2e'), np.float32) * dark * 0.35 * (1 - m[..., None])
        fx.add(img, fx.aura(m, t, col_in='#fff3c0', col_out='#ff5a1f', strength=0.7 + 0.6 * k / 35, height=140, seed=8))
        embers(img, t, seed=12, n=30, col='#ff9a3c', speed=(200, 380))
        img = fx.bloom(img, 0.75, 26, 0.4)
        hx, hy = TRACK[src]['head']
        z = 1.0 + 0.32 * ease_in_cubic(k / 35)
        img = cam(img, zoom=z, cx=hx, cy=hy + 120)
        fl = focus('pu', hx, hy + 120, 100 + k // 2, n=170, r_in=(420, 640), wid=(3, 16))
        fx.over_mask(img, hexc('#0d0612'), fl * min(1.0, k / 6) * 0.9)
        g = gogo_sprite()
        nvis = min(4, 1 + k // 5)
        for side, x in ((0, 120), (1, 960)):
            jx, jy = jitter(k, 5, 11 + side)
            sub_h = int(g.shape[0] * nvis / 4)
            part = g[:sub_h]
            blit(img, part, x + jx, 260 + sub_h / 2 + jy + side * 120, scale=0.92, angle=-6 if side == 0 else 6)
        img = shake(img, k, 3 + 5 * k / 35, seed=4)
        return img
    # --- close-up glasses glint (freeze f043, high-res crop) ---
    kk = k - 36
    cimg, cm = closeup()
    img = cimg.copy()
    bgd = 0.55
    img = img * (1 - bgd * (1 - cm[..., None])) + np.array(hexc('#120a2a'), np.float32) * bgd * (1 - cm[..., None])
    _glasses_glare(img, kk)
    z = 1.0 + 0.05 * ease_out_cubic(kk / 23)
    img = cam(img, zoom=z, cx=566, cy=790)
    fl = focus('cu', 655, 762, 200 + kk // 2, n=150, r_in=(520, 760), wid=(2, 12))
    fx.over_mask(img, (1, 1, 1), fl * 0.55)
    gx, gy = 566 + (655 - 566) * z, 790 + (762 - 790) * z
    u = (kk - 4) / 12
    if 0 <= u <= 1:
        s = math.sin(math.pi * u) ** 1.5 * 1.25
        blit(img, spr('sparkle', lambda: fx.sparkle(420)), gx, gy, scale=s, angle=u * 120, mode='add')
    if kk >= 6:
        kir = spr('kira', lambda: fx.text_sprite('キラーン', 'dela', 96, fill='#ffffff', strokes=((14, '#3fb8ff'),)))
        a = min(1.0, (kk - 6) / 3)
        blit(img, kir, 800, 560, angle=8, alpha=a, scale=0.9 + 0.1 * ease_out_back(min(1, (kk - 6) / 5)))
    draw_sub(img, "October... you're MINE.", kk, 7, 23)
    img = fx.vignette(img, 0.45)
    return flash(img, max(0.0, 0.7 - kk / 3))


def _glasses_glare(img, kk):
    """Opaque anime lens glare over both lenses of the close-up (classic megane shine)."""
    def build():
        lay = np.zeros((H, W), np.float32)
        for (x0, y0, x1, y1) in ((368, 752, 512, 822), (556, 748, 676, 814)):
            cv2.ellipse(lay, ((x0 + x1) // 2, (y0 + y1) // 2), ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 2.0)
        xx, yy = fx.grid()
        stripe = 0.82 + 0.18 * np.sin((xx + yy) * 0.09)
        return lay, stripe.astype(np.float32)
    lay, stripe = spr('glare', build)
    a = min(1.0, max(0.0, (kk - 2) / 4)) * 0.92
    col = np.dstack([stripe * 0.95, stripe * 0.98, stripe * 1.0])
    img[:] = img * (1 - lay[..., None] * a) + col * lay[..., None] * a


def tunnel_bg(k):
    def build():
        xx, yy = fx.grid()
        r = np.sqrt((xx - 540) ** 2 + (yy - 980) ** 2)
        u = np.clip(r / 1200, 0, 1)[..., None]
        return (np.array(hexc('#3a2a8a'), np.float32) * (1 - u) + np.array(hexc('#07051a'), np.float32) * u).astype(np.float32)
    img = spr('tunnel', build).copy()
    fl = focus('tun', 540, 980, 300 + k // 2, n=190, r_in=(260, 520), wid=(3, 18))
    fx.over_mask(img, (1, 1, 1), fl * 0.9)
    st = spr(('tstreak', k // 2 % 4), lambda: fx.streaks(500 + k // 2 % 4, angle_deg=-90, n=50, length=(300, 900), wid=(2, 6)))
    fx.over_mask(img, hexc('#ff4a6a'), st * 0.35)
    return img


def s_charge(k):
    t = k / FPS
    src = 49 + k // 3
    bg = tunnel_bg(k)
    lay = cutout_layer(src, 7, 0)
    cx, cy = TRACK[src]['center']
    img = place(bg, lay, scale=1.0 + 0.10 * ease_in_cubic(k / 35), ax=cx, ay=cy)
    fx.add(img, fx.aura(mask(src), t, col_in='#ffffff', col_out='#ff3a2a', strength=0.6, height=60, seed=4) * 0.8)
    hiss = spr('hissatsu', lambda: fx.text_sprite('必殺', 'brush', 250, fill='#ffffff', strokes=((26, '#e8202a'), (14, '#16121f'))))
    sc = stamp_scale(k, 2, dur=5, start=2.4)
    if sc:
        jx, jy = jitter(k, 4, 21)
        blit(img, hiss, 270 + jx, 330 + jy, scale=sc, angle=10)
    kick = spr('kick', lambda: fx.text_sprite('十月キック!!', 'dela', 132, grad=('#fff6b0', '#ff8a00'),
                                              strokes=((30, '#ffffff'), (17, '#16121f'))))
    sc = stamp_scale(k, 10, dur=5, start=2.6)
    if sc:
        jx, jy = jitter(k, 6, 22)
        blit(img, kick, 540 + jx, 1480 + jy, scale=sc * fit(kick, 1040), angle=-6)
    draw_sub(img, 'Special move: OCTOBER KICK!!', k, 10, 35, y=1730)
    img = flash(img, max(0.0, 0.6 - k / 4))
    return shake(img, k, 4 + 10 * k / 35, seed=6, rot=0.6)


def s_gate_hit(k):
    t = k / FPS
    if k < 14:
        g = gate_close()
        img = np.zeros((H, W, 3), np.float32)
        img = img * (1 - g[..., 3:4]) + g[..., :3]
        intensity = 0.7 + 0.9 * (k / 13) + 0.2 * math.sin(k * 2.1)
        fx.screen(img, gate.seam_glow(intensity, col='#ff8a3c'))
        if k >= 6:
            cr = spr('crack1', lambda: fx.crack_lines(gate.LOCK[0], gate.LOCK[1], 31, n=10, length=(120, 420), width=6))
            dm = np.zeros((H, W), np.float32); dm[gate.DY0:gate.DY1, gate.DX0:gate.DX1] = 1
            fx.over_mask(img, hexc('#0d0608'), cr * dm)
            glowc = cv2.GaussianBlur(cr * dm, (0, 0), 6)
            fx.add(img, glowc[..., None] * np.array(hexc('#ff9a3c'), np.float32), 1.2)
        rng = np.random.default_rng(77)
        for _ in range(40):   # dust falling from the lintel
            x = rng.uniform(150, 930); v = rng.uniform(300, 700); y0 = 430 + rng.uniform(-40, 0); st = rng.uniform(0, 0.3)
            if t > st:
                y = y0 + v * (t - st) + 400 * (t - st) ** 2
                cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 5)), (0.75, 0.68, 0.6), -1, cv2.LINE_AA)
        z = 1.0 + 0.05 * k / 13 + (0.03 if 6 <= k <= 7 else 0)
        img = cam(img, zoom=z, cx=540, cy=1100)
        if 6 <= k <= 13:
            don = spr('don', lambda: fx.text_sprite('ドンッ!', 'dela', 150, fill='#ffffff', strokes=((26, '#16121f'), (12, '#ff4a2a'))))
            sc = stamp_scale(k, 6, dur=3, start=1.8)
            blit(img, don, 300, 1490, scale=sc, angle=10)
        return shake(img, k, 2 + 9 * k / 13 + (14 if 6 <= k <= 8 else 0), seed=8, rot=0.4)
    if k == 14:
        return np.ones((H, W, 3), np.float32)
    if k == 15:
        g = gate_close()
        img = np.zeros((H, W, 3), np.float32) * (1 - g[..., 3:4]) + g[..., :3]
        img = fx.impact_bw(img, 0.35, invert=True)
        cr = spr('crack2', lambda: fx.crack_lines(gate.LOCK[0], gate.LOCK[1], 32, n=18, length=(400, 1400), width=11))
        fx.over_mask(img, (1, 1, 1), cr)
        return shake(img, k, 30, seed=9, rot=2)
    if k in (16, 17):
        src = 64
        m = mask(src)
        a = anime(src)
        e = fx.xdog_edges(a)
        lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
        ink = e * smoothstep(0.08, 0.16, lum)          # 1 = paper, 0 = ink
        cx, cy = TRACK[src]['center']
        if k == 16:
            img = np.zeros((H, W, 3), np.float32) + np.array(hexc('#e8102a'), np.float32)
            fl = focus('imp', cx, cy, 401, n=200, r_in=(240, 460), wid=(4, 22))
            fx.over_mask(img, (0, 0, 0), fl)
            fig = ink[..., None] * np.array([1, 0.96, 0.9], np.float32)
        else:
            img = np.zeros((H, W, 3), np.float32)
            fl = focus('imp2', cx, cy, 402, n=200, r_in=(240, 460), wid=(4, 22))
            fx.over_mask(img, hexc('#e8102a'), fl)
            fig = (1 - ink)[..., None] * np.ones(3, np.float32)
        img = img * (1 - m[..., None]) + fig * m[..., None]
        line, _ = fx.outline(m, 7, 0)
        fx.over_mask(img, (0, 0, 0) if k == 16 else (1, 1, 1), line)
        return shake(img, k, 26, seed=10, rot=2)
    # ---- reveal: doors blown toward camera, Yt mid-kick behind ----
    kk = k - 18
    tt = kk / FPS
    src = min(67, 62 + kk // 6)
    img = anime(src).copy()
    cx, cy = TRACK[src]['center']
    rays = fx.god_rays(600, 250, tt * 0.8, n=14, spread=0.38, falloff=1300)
    fx.screen(img, rays[..., None] * np.array(hexc('#fff0c0'), np.float32))
    img = cam(img, zoom=1.0 + 0.09 * ease_out_cubic(kk / 35), cx=cx, cy=cy)
    fa = max(0.0, 1 - kk / 22)
    fl = focus('rev', cx, cy - 100, 500 + kk // 2, n=170, r_in=(380, 620), wid=(3, 15))
    fx.over_mask(img, (1, 1, 1), fl * 0.75 * fa)
    # gate frame pushes out past the camera
    u = ease_in_cubic(min(1.0, kk / 16))
    gf = gate_close(with_doors=False)
    if u < 0.999:
        z = 1.0 + 1.6 * u
        M = cv2.getRotationMatrix2D((540, 1060), 0, z)
        wp = cv2.warpAffine(gf, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        img = img * (1 - wp[..., 3:4]) + wp[..., :3]
    # doors
    ds = door_sprites()
    v = ease_out_cubic(min(1.0, kk / 13))
    a_d = 1.0 if v < 0.75 else max(0.0, 1 - (v - 0.75) / 0.25)
    for side, sgn in (('l', -1), ('r', 1)):
        x0, y0 = ds[side + '_c']
        blit(img, ds[side], x0 + sgn * 1350 * v, y0 - 160 * v, scale=1 + 1.2 * v, angle=sgn * 22 * v, alpha=a_d)
    # padlock flies at the camera
    if kk < 9:
        w_ = ease_out_cubic(kk / 8)
        blit(img, G['lock'], 540 + 950 * w_, 1175 - 1500 * w_, scale=1 + 2.6 * w_, angle=540 * w_)
    # debris + dust
    for d in debris_list():
        x = d['x0'] + d['vx'] * tt; y = d['y0'] + d['vy'] * tt + 900 * tt * tt
        s = 1 + d['z'] * tt * 3
        if -200 < x < W + 200 and -200 < y < H + 200:
            blit(img, d['s'], (x - 540) * (1 + d['z'] * tt) + 540, (y - 1100) * (1 + d['z'] * tt) + 1100, scale=s, angle=d['spin'] * tt)
    rng = np.random.default_rng(55)
    blob = spr('blob', lambda: fx.soft_blob(256, '#e8d8c0', 0.55))
    for _ in range(10):
        ang = rng.uniform(0, 2 * math.pi); sp_ = rng.uniform(500, 1100)
        x = 540 + math.cos(ang) * (220 + sp_ * tt); y = 1150 + math.sin(ang) * (220 + sp_ * tt)
        blit(img, blob, x, y, scale=0.8 + 2.2 * tt, alpha=max(0.0, 0.45 - tt * 0.6))
    # big SFX
    dg = spr('dogon', lambda: fx.text_sprite('ドゴォォン!!', 'dela', 160, grad=('#fff6b0', '#ff7a00'),
                                             strokes=((34, '#ffffff'), (19, '#16121f'))))
    sc = stamp_scale(kk, 0, dur=6, start=2.8)
    if kk <= 32:
        amp = 10 * max(0.0, 1 - kk / 20)
        jx, jy = jitter(kk, amp, 31)
        a = 1.0 if kk < 28 else (32 - kk) / 4
        blit(img, dg, 540 + jx, 1470 + jy, scale=sc * fit(dg, 1020), angle=-6, alpha=a)
    img = flash(img, max(0.0, 0.75 - kk / 4))
    img = fx.bloom(img, 0.78, 28, 0.3)
    amp = 46 * math.exp(-kk / 6)
    return shake(img, kk, amp, seed=12, rot=1.6 * math.exp(-kk / 6))


def s_title_slam(k):
    t = k / FPS
    src = 66
    bg = fx.sunburst(760, 1000, k * 0.012, n=20)
    ht = spr('ht2', lambda: fx.halftone(H, W, cell=20, angle=math.radians(15)))
    bg = bg * (1 - 0.12 * ht[..., None])
    lay = spr(('cut2', src), lambda: cutout_layer(src, 7, 16))
    sc = 0.78 + 0.03 * ease_out_cubic(k / 89)
    img = place(bg, lay, scale=sc, ax=W, ay=H)
    leaves(img, t, seed=7, n=20, wind=-160, scale=(0.4, 1.0))
    sparkles(img, t, seed=3, n=10, region=(40, 1040, 40, 640))
    for word, k0, y in (('OCTOBER', 4, 205), ('SALES', 10, 440)):
        s = spr(('title', word), lambda word=word: fx.text_sprite(word, 'anton', 236, grad=('#fffbe0', '#ffb300'),
                                                                   strokes=((36, '#ffffff'), (20, '#7a0f12')),
                                                                   shadow=(12, 14, '#3a0608'), tracking=6))
        sc_ = stamp_scale(k, k0, dur=5, start=3.0)
        if sc_:
            jx, jy = jitter(k, 8 if k - k0 < 6 else 0, 40 + k0)
            blit(img, s, 540 + jx, y + jy, scale=sc_ * fit(s, 1040), angle=-3)
    kai = spr('kaimon', lambda: fx.vtext_sprite('開門', 'dela', 210, fill='#ff3a2a',
                                                 strokes=((32, '#ffffff'), (18, '#16121f')), grad=('#ff6a4a', '#c40018')))
    ex = spr('kaimon_ex', lambda: fx.text_sprite('!!', 'dela', 170, grad=('#ff6a4a', '#c40018'), strokes=((32, '#ffffff'), (18, '#16121f'))))
    sc_ = stamp_scale(k, 22, dur=5, start=2.6)
    if sc_:
        jx, jy = jitter(k, 7 if k < 30 else 0, 50)
        blit(img, kai, 175 + jx, 1060 + jy, scale=sc_, angle=6)
        blit(img, ex, 185 + jx, 1430 + jy, scale=sc_ * 0.8, angle=6)
    draw_sub(img, 'THE GATES ARE OPEN!!', k, 22, 89, y=1730)
    hit = max(0.0, 0.8 - k / 4) + (max(0.0, 0.3 - (k - 4) / 6) if k >= 4 else 0) + (max(0.0, 0.3 - (k - 10) / 6) if k >= 10 else 0) \
        + (max(0.0, 0.4 - (k - 22) / 6) if k >= 22 else 0)
    img = flash(img, min(1.0, hit))
    img = cam(img, zoom=1.0 + 0.03 * k / 89, cx=540, cy=900)
    amp = 0
    for k0, a0 in ((4, 14), (10, 14), (22, 20)):
        if k >= k0:
            amp += a0 * math.exp(-(k - k0) / 3)
    return shake(img, k, amp, seed=14, rot=0.5)


def s_walk_off(k):
    t = k / FPS
    src = 67 + int(round(k * (95 - 67) / 83))
    img = anime(src).copy()
    img = fx.grade(img, sat=1.1, contrast=1.04, warm=0.10)
    rays = fx.god_rays(260, 120, t * 0.5, n=12, spread=0.45, falloff=1400)
    fx.screen(img, rays[..., None] * np.array(hexc('#ffe2a8'), np.float32))
    img = fx.bloom(img, 0.68, 32, 0.6)
    img = cam(img, zoom=1.05 - 0.05 * ease_in_out(k / 83), cx=540, cy=900)
    leaves(img, t, seed=17, n=14, scale=(0.4, 1.0))
    sparkles(img, t, seed=8, n=8, region=(40, 1040, 80, 1200), size=(30, 80))
    draw_sub(img, "Now let's go sell some cars.", k, 10, 80)
    return fx.vignette(img, 0.35)


def s_tsuzuku(k):
    t = k / FPS
    src = 95
    img = anime(src).copy()
    img = fx.grade(img, sat=1.1, contrast=1.04, warm=0.10)
    lum = img @ np.array([0.299, 0.587, 0.114], np.float32)
    sep = np.dstack([lum * 1.07 + 0.05, lum * 0.93 + 0.03, lum * 0.72])
    u = ease_out_cubic(k / 6)
    img = np.clip(img * (1 - u) + sep * u, 0, 1)
    ht = spr('ht3', lambda: fx.halftone(H, W, cell=14, angle=math.radians(45)))
    img = img * (1 - 0.10 * u * ht[..., None])
    img = cam(img, zoom=1.0 + 0.02 * k / 65, cx=540, cy=900)
    tz = spr('tsuzuku', lambda: fx.text_sprite('つづく', 'brush', 230, fill='#1a0f14', strokes=((18, '#fff6e6'),)))
    if k >= 6:
        a = min(1.0, (k - 6) / 5)
        blit(img, tz, 740 - 40 * (1 - a), 1530, alpha=a, angle=-4)
    tbc = spr('tbc', lambda: fx.text_sprite('TO BE CONTINUED...', 'bangers', 72, fill='#ffffff', strokes=((9, '#1a0f14'),), tracking=4))
    if k >= 14:
        a = min(1.0, (k - 14) / 5)
        blit(img, tbc, 700, 1735, alpha=a, angle=-4)
    img = fx.vignette(img, 0.45)
    img = flash(img, max(0.0, 0.6 - k / 4))
    fade = 1.0 - float(np.clip((k - 52) / 13, 0, 1))
    return img * fade


SCENE_FN = {name: globals()['s_' + name] for name, _ in SCENES}


def render(gf):
    for name, n in SCENES:
        if START[name] <= gf < START[name] + n:
            img = SCENE_FN[name](gf - START[name])
            return np.clip(img, 0, 1)
    raise ValueError(gf)


def to_bgr8(img):
    return cv2.cvtColor((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        os.makedirs(f'{S}/test_frames', exist_ok=True)
        for a in sys.argv[2:]:
            if ':' in a:
                name, lf = a.split(':'); gf = START[name] + int(lf)
            else:
                gf = int(a)
            cv2.imwrite(f'{S}/test_frames/t{gf:04d}.jpg', to_bgr8(render(gf)), [cv2.IMWRITE_JPEG_QUALITY, 88])
            print('wrote', gf, flush=True)
    elif mode == 'range':
        os.makedirs(OUT, exist_ok=True)
        a, b = int(sys.argv[2]), int(sys.argv[3])
        for gf in range(a, min(b, TOTAL)):
            p = f'{OUT}/f{gf:04d}.png'
            if not os.path.exists(p):
                cv2.imwrite(p, to_bgr8(render(gf)), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            print('frame', gf, flush=True)
