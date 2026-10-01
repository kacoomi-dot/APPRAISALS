"""Synthesized anime soundtrack + SFX for the short, aligned to timeline.py."""
import sys, os, wave
import numpy as np
from scipy import signal
from timeline import FPS, TOTAL, sec

SR = 48000
DUR = TOTAL / FPS + 0.3
N = int(DUR * SR)
rng = np.random.default_rng(1234)
music = np.zeros((N, 2)); sfx = np.zeros((N, 2)); send = np.zeros((N, 2))


def tt(d):
    return np.arange(int(d * SR)) / SR


def noise(n):
    return rng.standard_normal(n)


def lp(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos'), x)


def hp(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, fc, 'high', fs=SR, output='sos'), x)


def bp(x, f0, f1, order=3):
    return signal.sosfilt(signal.butter(order, [f0, min(f1, SR * 0.45)], 'band', fs=SR, output='sos'), x)


def put(bus, x, t, gain=1.0, pan=0.0, rev=0.0):
    i = int(t * SR)
    if i >= N:
        return
    n = min(len(x), N - i)
    if x.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4) * 1.414, np.sin((pan + 1) * np.pi / 4) * 1.414
        bus[i:i + n, 0] += x[:n] * gain * l; bus[i:i + n, 1] += x[:n] * gain * r
        if rev:
            send[i:i + n, 0] += x[:n] * gain * l * rev; send[i:i + n, 1] += x[:n] * gain * r * rev
    else:
        bus[i:i + n] += x[:n] * gain
        if rev:
            send[i:i + n] += x[:n] * gain * rev


def adsr(n, a=0.005, d=0.1, s=0.7, r=0.2, hold=None):
    a_, d_, r_ = int(a * SR), int(d * SR), int(r * SR)
    h = max(0, (n - a_ - d_ - r_) if hold is None else int(hold * SR))
    e = np.concatenate([np.linspace(0, 1, max(1, a_)), np.linspace(1, s, max(1, d_)), np.full(h, s), np.linspace(s, 0, max(1, r_))])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def saw(f, t, det=0.0):
    ph = (f * (1 + det)) * t + rng.uniform(0, 1)
    return 2 * (ph - np.floor(ph)) - 1


def note(name):
    names = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    p, o = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((names[p] + 12 * (o - 4)) / 12)


# ------------------------------------------------------------------ instruments
def taiko(f0=110, f1=52, tau=0.38, d=1.4):
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / tau) + 0.3 * np.sin(1.53 * ph) * np.exp(-t / (tau * 0.35))
    hit = lp(noise(len(t)), 1200) * np.exp(-t / 0.025) * 0.9
    return np.tanh(2.0 * (body + hit)) * 0.9


def kick():
    t = tt(0.35)
    f = 48 + 110 * np.exp(-t / 0.03)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    x += hp(noise(len(t)), 3000) * np.exp(-t / 0.004) * 0.3
    return np.tanh(1.5 * x)


def snare():
    t = tt(0.3)
    x = bp(noise(len(t)), 1500, 7000) * np.exp(-t / 0.11) * 0.8
    x += np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.07) * 0.6
    return x


def hat(open_=False):
    t = tt(0.4 if open_ else 0.08)
    return hp(noise(len(t)), 7500) * np.exp(-t / (0.18 if open_ else 0.025)) * 0.35


def crash(d=3.0, g=0.6):
    t = tt(d)
    x = hp(noise(len(t)), 3500) * np.exp(-t / 0.9)
    for f in (3150, 4720, 6310, 8900, 5470):
        x += 0.08 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * np.exp(-t / 0.7)
    return x * g


def gong(f=96, d=4.5):
    t = tt(d)
    x = np.zeros_like(t)
    for k, (m, a, tau) in enumerate([(1, 1, 2.2), (1.47, 0.6, 1.8), (2.02, 0.5, 1.4), (2.51, 0.35, 1.2), (3.2, 0.25, 0.9), (4.13, 0.15, 0.7)]):
        x += a * np.sin(2 * np.pi * f * m * t * (1 + 0.002 * np.sin(2 * np.pi * 3 * t)) + k) * np.exp(-t / tau)
    x *= np.minimum(1, t / 0.01)
    return x * 0.35


def supersaw(freqs, d, cutoff=2500, a=0.01, dec=0.3, s=0.6, r=0.3, voices=5, spread=0.012, cut_env=None):
    t = tt(d)
    x = np.zeros_like(t)
    for f in freqs:
        for v in range(voices):
            det = (v - (voices - 1) / 2) * spread / max(1, voices - 1) * 2
            x += saw(f, t, det)
    x /= (len(freqs) * voices) ** 0.5 * 2
    if cut_env is None:
        x = lp(x, cutoff)
    else:
        lo = lp(x, cut_env[1]); hi = lp(x, cut_env[0])
        m = np.exp(-t / cut_env[2])
        x = hi * m + lo * (1 - m)
    return x * adsr(len(t), a, dec, s, r)


def brass_stab(freqs, d=0.7):
    return supersaw(freqs, d, a=0.004, dec=0.25, s=0.45, r=0.25, cut_env=(7000, 1400, 0.12)) * 1.6


def orch_hit(freqs, d=1.6):
    x = brass_stab(freqs + [f * 2 for f in freqs] + [freqs[0] / 2], d)
    t = tt(d)
    x += lp(noise(len(t)), 5000) * np.exp(-t / 0.05) * 0.5
    x += np.pad(taiko(90, 45, 0.5, d), (0, 0))[:len(t)] * 0.8
    return x


def whoosh(d=0.8, f0=300, f1=3000, peak=0.6, g=0.7):
    n = int(d * SR)
    x = noise(n)
    f, tm, Z = signal.stft(x, fs=SR, nperseg=1024)
    fc = f0 * (f1 / f0) ** np.clip(tm / d, 0, 1)
    lf = np.log(np.maximum(f, 20))[:, None]
    Z *= np.exp(-((lf - np.log(fc)[None, :]) ** 2) / (2 * 0.35 ** 2))
    _, y = signal.istft(Z, fs=SR, nperseg=1024)
    y = np.pad(y, (0, max(0, n - len(y))))[:n]
    u = np.arange(len(y)) / max(1, len(y))
    env = np.where(u < peak, (u / peak) ** 2, ((1 - u) / (1 - peak)) ** 1.5)
    y = y * env
    return y / (np.abs(y).max() + 1e-9) * g


def riser(d=1.2):
    t = tt(d)
    x = whoosh(d, 250, 7000, 0.97, 0.6)
    f = 180 * (1400 / 180) ** (t / d)
    x[:len(t)] += 0.18 * np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / d) ** 2
    return x


def sub_boom(d=2.6, f0=75, f1=24):
    t = tt(d)
    f = f1 + (f0 - f1) * np.exp(-t / 0.35)
    return np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9))


def clang(f=170, d=2.8):
    t = tt(d)
    x = np.zeros_like(t)
    for m, a, tau in [(1, 0.6, 1.4), (2.41, 0.5, 1.1), (3.87, 0.4, 0.8), (5.53, 0.3, 0.6), (7.9, 0.2, 0.4), (11.3, 0.15, 0.25)]:
        x += a * np.sin(2 * np.pi * f * m * t + rng.uniform(0, 6)) * np.exp(-t / tau)
    x += hp(noise(len(t)), 2000) * np.exp(-t / 0.03) * 0.6
    return x * 0.45


def crack(d=0.9):
    t = tt(d)
    x = hp(noise(len(t)), 900) * np.exp(-t / 0.06)
    for _ in range(14):
        i = int(rng.uniform(0.0, 0.5) * SR); n = int(0.012 * SR)
        if i + n < len(x):
            x[i:i + n] += bp(noise(n), 1500, 6000) * rng.uniform(0.3, 0.9)
    return x * 0.7


def debris(d=1.8, count=40):
    x = np.zeros(int(d * SR))
    for _ in range(count):
        tpos = rng.uniform(0.05, d - 0.1) ** 1.6 / d ** 0.6
        i = int(tpos * SR); n = int(rng.uniform(0.01, 0.04) * SR)
        if i + n < len(x):
            seg = bp(noise(n), rng.uniform(400, 1500), 7000) * np.exp(-np.arange(n) / (n / 4))
            x[i:i + n] += seg * rng.uniform(0.15, 0.5) * (1 - tpos / d)
    return x


def chains(d=1.0, count=18):
    x = np.zeros(int(d * SR))
    for _ in range(count):
        i = int(rng.uniform(0, d * 0.8) * SR); n = int(0.09 * SR)
        if i + n < len(x):
            tn = np.arange(n) / SR
            f = rng.uniform(2200, 5200)
            x[i:i + n] += (np.sin(2 * np.pi * f * tn) + 0.5 * np.sin(2 * np.pi * f * 1.47 * tn)) * np.exp(-tn / 0.02) * rng.uniform(0.1, 0.3)
    return x


def ting():
    t = tt(1.6)
    x = np.zeros_like(t)
    for f, a, tau in [(2637, 0.6, 0.9), (3951, 0.45, 0.6), (5274, 0.3, 0.45), (7040, 0.2, 0.3)]:
        x += a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    sweep = np.sin(2 * np.pi * np.cumsum(4000 + 5000 * t[:int(0.15 * SR)] / 0.15) / SR) * np.linspace(0.3, 0, int(0.15 * SR))
    x[:len(sweep)] += sweep * 0.5
    return x * 0.5


def heartbeat():
    t = tt(0.6)
    def thump(tau, f):
        return np.sin(2 * np.pi * f * t) * np.exp(-t / tau) * np.minimum(1, t / 0.005)
    x = thump(0.08, 52)
    x2 = np.zeros_like(x); i = int(0.22 * SR)
    x2[i:] = thump(0.07, 58)[:len(x) - i] * 0.8
    return np.tanh(2 * (x + x2))


def drone(d):
    t = tt(d)
    x = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in [(36.7, 0.5, 0), (55.0, 0.35, 1), (73.4, 0.3, 2), (87.3, 0.15, 3)])
    x *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.7 * t)
    x += lp(noise(len(t)), 220) * 0.25
    return x * np.minimum(1, t / 0.6)


def wind(d):
    t = tt(d)
    x = lp(np.cumsum(noise(len(t))) * 0.02, 700)
    x = hp(x, 80)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t + 1)
    return x / (np.abs(x).max() + 1e-9) * 0.4


def hum(d, f0=55, f1=110, g=0.25):
    t = tt(d)
    f = f0 * (f1 / f0) ** (t / d)
    x = lp(np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)), 500)
    crackle = lp(noise(len(t)), 3000) * (rng.random(len(t)) < 0.002)
    x = x * (0.7 + 0.3 * np.sin(2 * np.pi * 9 * t)) + crackle * 2
    return x * g * np.minimum(1, t / 0.15) * np.minimum(1, (d - t) / 0.1)


def pop():
    t = tt(0.12)
    f = 700 + 900 * t / 0.12
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.04) * 0.5


def lead(freq, d, vib=True):
    t = tt(d)
    f = freq * (1 + (0.006 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.25) if vib else 0))
    ph = np.cumsum(f) / SR
    x = 0.6 * (2 * (ph - np.floor(ph)) - 1) + 0.4 * np.sign(np.sin(2 * np.pi * ph))
    x = lp(x, 3800)
    return x * adsr(len(t), 0.01, 0.12, 0.75, 0.08) * 0.35


# ------------------------------------------------------------------ score
# Cold open: ominous drone, wind, taiko
put(music, drone(3.25), 0.0, 0.55, rev=0.2)
put(sfx, wind(3.2), 0.0, 0.5)
put(sfx, wind(2.0), sec('walk_off', 0), 0.25)
put(music, taiko(80, 40, 0.6, 2.0), 0.08, 1.0, rev=0.35)
put(music, taiko(80, 40, 0.6, 2.0), sec('cold_open', 53), 0.9, rev=0.35)
put(sfx, sub_boom(2.0, 45, 22) * 0.35, sec('cold_open', 53), 0.6)

# Episode title stinger
D_min = [note('D3'), note('F3'), note('A3')]
put(music, orch_hit(D_min, 1.8), sec('ep_title'), 0.9, rev=0.4)
put(music, gong(92, 4.0), sec('ep_title'), 0.8, rev=0.3)

# Yt intro: aura hum + whoosh, pulse bass begins
put(sfx, whoosh(0.7, 200, 4000, 0.7), sec('yt_intro') - 0.35, 0.6)
put(sfx, hp(hum(3.3, 55, 82, 0.14), 70), sec('yt_intro'), 1.0)
BEAT = 60 / 150
t0 = sec('yt_intro')
pulse_end = sec('power_up', 36)
k = 0
while t0 + k * BEAT / 2 < pulse_end:
    tp = t0 + k * BEAT / 2
    prog = (tp - t0) / (pulse_end - t0)
    x = supersaw([note('D2')], BEAT / 2 * 0.9, cutoff=300 + 1800 * prog, a=0.003, dec=0.1, s=0.5, r=0.05, voices=3)
    put(music, x, tp, 0.55 + 0.3 * prog)
    if k % 4 == 0:
        put(music, taiko(120, 60, 0.25, 0.8), tp, 0.35 + 0.4 * prog, rev=0.2)
    k += 1

# Character card
put(sfx, whoosh(0.5, 400, 6000, 0.5), sec('char_card') - 0.1, 0.7)
put(sfx, crash(1.2, 0.35), sec('char_card', 4), 0.5, rev=0.2)
put(music, brass_stab([note('D4'), note('F#4'), note('A4')], 0.5), sec('char_card', 5), 0.6, rev=0.25)
put(music, taiko(130, 60, 0.3, 1.0), sec('char_card', 15), 1.0, rev=0.25)
put(music, taiko(130, 60, 0.3, 1.0), sec('char_card', 17), 0.9, rev=0.25)
put(sfx, snare(), sec('char_card', 15), 0.6)
put(sfx, pop(), sec('char_card', 24), 0.5)

# Power-up: riser + heartbeat + rumble
put(sfx, riser(1.2), sec('power_up'), 0.55)
put(sfx, hum(1.2, 70, 140, 0.3), sec('power_up'), 1.0)
for j in range(3):
    put(sfx, heartbeat(), sec('power_up') + 0.05 + j * 0.42, 0.7)
put(sfx, sub_boom(1.4, 40, 25) * 0.6, sec('power_up'), 0.5)
# close-up: near silence, the glint ting
put(sfx, ting(), sec('power_up', 36 + 8), 0.8, pan=0.2, rev=0.5)
put(music, supersaw([note('D4'), note('A4'), note('E5')], 0.8, cutoff=1600, a=0.3, dec=0.2, s=0.6, r=0.3), sec('power_up', 40), 0.25, rev=0.4)

# Charge: whoosh + accelerating taiko roll + hits
put(sfx, whoosh(0.9, 300, 5000, 0.4), sec('charge') - 0.05, 0.8)
put(music, orch_hit([note('D3'), note('F3'), note('A3')], 1.0), sec('charge', 2), 0.6, rev=0.3)
put(music, orch_hit([note('E3'), note('G3'), note('B3')], 1.0), sec('charge', 10), 0.7, rev=0.3)
tr = sec('charge'); end = sec('gate_hit', 10)
tp = tr; gap = 0.16
while tp < end:
    prog = (tp - tr) / (end - tr)
    put(music, taiko(140, 70, 0.18, 0.5), tp, 0.35 + 0.5 * prog, pan=0.3 * np.sin(tp * 17), rev=0.15)
    tp += gap; gap = max(0.05, gap * 0.94)
put(sfx, riser(1.6), sec('charge') + 0.1, 0.4)

# Gate: rumble, thud, silence, IMPACT
put(sfx, sub_boom(1.0, 40, 30) * 0.5, sec('gate_hit'), 0.6)
put(sfx, taiko(70, 35, 0.4, 1.0), sec('gate_hit', 6), 1.0, rev=0.2)
put(sfx, crack(0.4) * 0.5, sec('gate_hit', 6), 0.6)
put(sfx, debris(0.4, 8), sec('gate_hit', 6), 0.5)
T = sec('gate_hit', 14)
put(sfx, sub_boom(2.8, 80, 22), T, 1.15)
put(sfx, crash(3.0, 0.9), T, 0.9, rev=0.4)
put(sfx, clang(165, 2.8), T, 0.9, pan=-0.15, rev=0.45)
put(sfx, clang(240, 2.0), T + 0.02, 0.6, pan=0.2, rev=0.45)
put(sfx, crack(0.9), T, 1.0, rev=0.3)
put(sfx, lp(noise(int(0.5 * SR)), 2500) * np.exp(-tt(0.5) / 0.12) * 0.9, T, 1.0)
put(sfx, debris(2.0, 60), T + 0.05, 0.9, pan=-0.3)
put(sfx, debris(2.0, 60), T + 0.08, 0.9, pan=0.3)
put(sfx, chains(1.2, 22), T + 0.05, 0.9, rev=0.3)
put(sfx, whoosh(0.5, 2000, 300, 0.2), T + 0.03, 0.6, pan=0.6)
put(music, orch_hit([note('D3'), note('A3'), note('D4'), note('F#4')], 2.0), T, 0.8, rev=0.5)

# Title slam
for lf, chord in ((4, ['D4', 'F#4', 'A4']), (10, ['E4', 'G#4', 'B4'])):
    put(music, brass_stab([note(n) for n in chord], 0.6), sec('title_slam', lf), 0.9, rev=0.3)
    put(music, taiko(120, 55, 0.35, 1.2), sec('title_slam', lf), 0.9, rev=0.25)
    put(sfx, crash(1.0, 0.25), sec('title_slam', lf), 0.5)
K = sec('title_slam', 22)
put(music, orch_hit([note('A3'), note('C#4'), note('E4')], 1.4), K, 0.9, rev=0.4)
put(sfx, crash(3.0, 0.8), K, 0.8, rev=0.3)
put(music, gong(110, 3.5), K, 0.6, rev=0.3)

# Theme: royal-road progression (IVmaj7 - V7 - iiim7 - vim7) in D major @150bpm
TH = sec('title_slam', 30)
BAR = 4 * BEAT
chords = [['G3', 'B3', 'D4', 'F#4'], ['A3', 'C#4', 'E4', 'G4'], ['F#3', 'A3', 'C#4', 'E4']]
roots = ['G2', 'A2', 'F#2']
for b, (ch, rt) in enumerate(zip(chords, roots)):
    tb = TH + b * BAR
    put(music, supersaw([note(n) for n in ch], BAR, cutoff=2600, a=0.02, dec=0.3, s=0.7, r=0.15), tb, 0.8, rev=0.25)
    for e in range(8):
        f = note(rt) * (2 if e % 2 else 1)
        put(music, supersaw([f], BEAT / 2 * 0.85, cutoff=900, a=0.003, dec=0.08, s=0.6, r=0.04, voices=2), tb + e * BEAT / 2, 0.85)
    for e in range(8):
        tp = tb + e * BEAT / 2
        put(music, hat(open_=(e == 7)), tp, 0.9, pan=0.25)
        if e in (0, 3, 4):
            put(music, kick(), tp, 1.25)
        if e in (2, 6):
            put(music, snare(), tp, 1.0, rev=0.15)
    if b == 2:
        for s16 in range(4):
            put(music, snare(), tb + 3 * BEAT + s16 * BEAT / 4, 0.4 + 0.12 * s16, rev=0.15)
melody = [('F#5', 1), ('A5', 1), ('B5', 1.5), ('A5', 0.5),
          ('A5', 1), ('G5', 1), ('F#5', 1), ('E5', 1),
          ('F#5', 2), ('E5', 1), ('C#5', 1)]
tp = TH
for n_, beats in melody:
    put(music, lead(note(n_), beats * BEAT * 0.95), tp, 1.05, rev=0.3)
    tp += beats * BEAT
# final chord on つづく (Bm7) + sting
TZ = sec('tsuzuku')
put(music, supersaw([note(n) for n in ['B2', 'D3', 'F#3', 'A3', 'B3', 'D4']], 2.3, cutoff=2400, a=0.01, dec=0.6, s=0.5, r=0.9), TZ, 0.7, rev=0.4)
put(music, lead(note('F#5'), 1.9), TZ, 0.7, rev=0.4)
put(music, kick(), TZ, 1.0); put(music, crash(2.5, 0.6), TZ, 0.7, rev=0.3)
put(music, taiko(110, 50, 0.5, 1.6), TZ, 0.8, rev=0.3)

# ------------------------------------------------------------------ mix
ir_t = tt(1.8)
ir = np.stack([noise(len(ir_t)), noise(len(ir_t))], 1) * np.exp(-ir_t / 0.45)[:, None]
ir[:, 0] = lp(ir[:, 0], 5000); ir[:, 1] = lp(ir[:, 1], 5000)
ir /= np.sqrt((ir ** 2).sum(0, keepdims=True))
wet = np.stack([signal.fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1)
mixd = music * 0.8 + sfx * 1.0 + wet * 0.35
mixd = hp(mixd.T, 25, 2).T
fade_n = int(0.6 * SR)
mixd[-fade_n:] *= np.linspace(1, 0, fade_n)[:, None]
# broadband compressor (3:1 above -18 dB re peak, 5 ms attack / 120 ms release)
mixd = mixd / np.abs(mixd).max()
lvl = np.abs(mixd).max(1)
env = np.zeros_like(lvl); a_att, a_rel = np.exp(-1 / (0.005 * SR)), np.exp(-1 / (0.12 * SR))
e = 0.0
lvl_s = signal.lfilter([1 - a_rel], [1, -a_rel], lvl)       # smoothed release envelope
env = np.maximum(lvl_s, signal.lfilter([1 - a_att], [1, -a_att], lvl))
thr = 10 ** (-18 / 20)
gain = np.where(env > thr, (env / thr) ** (1 / 3 - 1), 1.0)
mixd = mixd * gain[:, None]
peak = np.abs(mixd).max()
mixd = mixd / peak * 1.5
mixd = np.tanh(mixd) / np.tanh(1.5) * 0.94
out = sys.argv[1] if len(sys.argv) > 1 else 'soundtrack.wav'
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mixd * 32767).astype(np.int16).tobytes())
print('wrote', out, f'{DUR:.2f}s', 'peak before norm', round(float(peak), 2))
