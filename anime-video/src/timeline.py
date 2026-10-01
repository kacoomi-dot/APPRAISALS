"""Shared scene timeline (frames @ 30 fps) for picture and sound."""
FPS = 30
SCENES = [
    ('cold_open', 96),    # the sealed gate, narration
    ('ep_title', 66),     # episode title card
    ('yt_intro', 30),     # Yt in ready stance, aura
    ('char_card', 69),    # freeze: YT 参上!!
    ('power_up', 60),     # menacing ゴゴゴ, point + glasses glint
    ('charge', 36),       # 必殺 十月キック!!
    ('gate_hit', 54),     # impact, doors blown off
    ('title_slam', 90),   # OCTOBER SALES 開門!!
    ('walk_off', 84),     # walk into the month
    ('tsuzuku', 66),      # つづく
]
START = {}
_t = 0
for _n, _d in SCENES:
    START[_n] = _t
    _t += _d
TOTAL = _t


def sec(scene, local_frame=0):
    return (START[scene] + local_frame) / FPS
