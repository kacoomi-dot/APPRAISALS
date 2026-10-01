"""AnimeGANv3 (Shinkai) stylization and U2Net human segmentation via onnxruntime."""
import os, functools
import numpy as np, cv2
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@functools.lru_cache(maxsize=None)
def _session(name):
    return ort.InferenceSession(os.path.join(HERE, 'models', name), providers=['CPUExecutionProvider'])


def anime(img):
    """BGR uint8 frame -> anime-styled BGR uint8 frame of the same size."""
    sess = _session('Shinkai_37.onnx')
    small = cv2.resize(img, (720, 1280), interpolation=cv2.INTER_AREA)
    small = cv2.edgePreservingFilter(small, flags=1, sigma_s=25, sigma_r=0.25)   # tame skin texture/stubble
    x = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
    y = sess.run(None, {sess.get_inputs()[0].name: x[None]})[0]
    y = np.clip((np.squeeze(y) + 1) * 127.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)


def mask(img):
    """BGR uint8 frame -> float32 person mask in [0, 1]."""
    sess = _session('u2net_human_seg.onnx')
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 320), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    x = (x / max(x.max(), 1e-6) - MEAN) / STD
    y = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return cv2.resize(y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
