#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Mehdi Rezaei Far
"""
Image Lab  -  Edge detection / Perspective correction / Restoration
===================================================================

GUI:   python image_lab.py [image]
CLI:   python image_lab.py in.jpg -o out.png -p "perspective:quad=auto,plate=1;unsharp:amount=1.5"
Batch: python image_lab.py in_folder -o out_folder --preset my.json [--auto-persp]
List:  python image_lab.py --list

Requirements:  pip install opencv-python numpy      (tkinter ships with Python)

Pipeline syntax (CLI):  step;step;...   step = name:key=value,key=value
Perspective quad:       quad=auto   or   quad=x1|y1|x2|y2|x3|y3|x4|y4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import sys
import threading
import time
import types
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAVE_TK = True
except ImportError:  # CLI still works without tkinter
    HAVE_TK = False
    tk = ttk = types.SimpleNamespace(Frame=object, Tk=object)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def ensure_bgr(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def odd(n):
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def imread(path):
    data = np.fromfile(path, dtype=np.uint8)  # unicode-safe on Windows
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Cannot encode image")
    buf.tofile(path)


# ----------------------------------------------------------------------------
# Operation registry
# ----------------------------------------------------------------------------


@dataclass
class P:
    key: str
    label: str
    kind: str  # int | float | bool | choice | str
    default: object
    lo: float = 0
    hi: float = 1
    step: float = 1
    choices: tuple = ()


@dataclass
class Op:
    name: str
    category: str
    label: str
    desc: str
    params: list
    func: object

    def defaults(self):
        return {p.key: p.default for p in self.params}


OPS: dict[str, Op] = {}


def register(name, category, label, desc, params):
    def deco(fn):
        OPS[name] = Op(name, category, label, desc, params, fn)
        return fn
    return deco


OUT_PARAMS = [
    P("output", "Output", "choice", "edges", choices=("edges", "overlay", "keep_color")),
    P("invert", "Invert (black on white)", "bool", False),
]


def finish_edges(img, edges, output, invert):
    e = edges.astype(np.uint8)
    base = ensure_bgr(img)
    a = (e.astype(np.float32) / 255.0)[..., None]
    if output == "overlay":
        red = np.zeros_like(base)
        red[...] = (0, 0, 255)
        out = base * (1 - a) + red * a
    elif output == "keep_color":
        out = base * a
    else:
        out = 255 - e if invert else e
    return ensure_bgr(out)


def scale_response(m, gain, thresh):
    ref = float(np.percentile(m, 99.7))
    ref = ref if ref > 1e-6 else 1.0
    out = np.clip(m / ref * 255.0 * gain, 0, 255)
    if thresh > 0:
        out[out < thresh] = 0
    return out.astype(np.uint8)


# ----------------------------------------------------------------------------
# EDGES
# ----------------------------------------------------------------------------


@register("canny", "Edges", "Canny", "Classic thin-edge detector. Auto mode picks thresholds from image median.", [
    P("auto", "Auto thresholds", "bool", True),
    P("sigma", "Auto sigma", "float", 0.33, 0.05, 1.0, 0.01),
    P("low", "Low threshold", "int", 50, 0, 500, 1),
    P("high", "High threshold", "int", 150, 0, 500, 1),
    P("blur", "Gaussian blur (odd)", "int", 5, 1, 21, 2),
    P("aperture", "Sobel aperture", "choice", "3", choices=("3", "5", "7")),
    P("l2", "L2 gradient", "bool", True),
    P("clahe", "CLAHE pre-boost (weak edges)", "bool", False),
    *OUT_PARAMS,
])
def canny(img, auto, sigma, low, high, blur, aperture, l2, clahe, output, invert):
    g = to_gray(img)
    if clahe:
        g = cv2.createCLAHE(3.0, (8, 8)).apply(g)
    if blur > 1:
        g = cv2.GaussianBlur(g, (odd(blur), odd(blur)), 0)
    if auto:
        v = float(np.median(g))
        low, high = int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))
    if high <= low:
        high = low + 1
    e = cv2.Canny(g, low, high, apertureSize=int(aperture), L2gradient=bool(l2))
    return finish_edges(img, e, output, invert)


@register("gradient", "Edges", "Gradient (Sobel/Scharr/Prewitt/Roberts)", "Gradient-magnitude edge maps with soft response.", [
    P("kind", "Kernel", "choice", "sobel", choices=("sobel", "scharr", "prewitt", "roberts")),
    P("ksize", "Sobel size", "choice", "3", choices=("1", "3", "5", "7")),
    P("blur", "Gaussian blur (odd)", "int", 3, 1, 15, 2),
    P("gain", "Gain", "float", 1.0, 0.2, 5.0, 0.05),
    P("thresh", "Cut below", "int", 0, 0, 255, 1),
    *OUT_PARAMS,
])
def gradient(img, kind, ksize, blur, gain, thresh, output, invert):
    g = to_gray(img)
    if blur > 1:
        g = cv2.GaussianBlur(g, (odd(blur), odd(blur)), 0)
    g = g.astype(np.float32)
    if kind == "sobel":
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=int(ksize))
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=int(ksize))
    elif kind == "scharr":
        gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    elif kind == "prewitt":
        kx = np.array([[1, 0, -1]] * 3, np.float32)
        gx, gy = cv2.filter2D(g, -1, kx), cv2.filter2D(g, -1, kx.T)
    else:
        kx = np.array([[1, 0], [0, -1]], np.float32)
        ky = np.array([[0, 1], [-1, 0]], np.float32)
        gx, gy = cv2.filter2D(g, -1, kx), cv2.filter2D(g, -1, ky)
    return finish_edges(img, scale_response(cv2.magnitude(gx, gy), gain, thresh), output, invert)


@register("log", "Edges", "Laplacian of Gaussian", "Blob/edge detector, good for fine details.", [
    P("sigma", "Sigma", "float", 1.5, 0.5, 6.0, 0.1),
    P("ksize", "Laplacian size", "choice", "3", choices=("1", "3", "5", "7")),
    P("gain", "Gain", "float", 1.0, 0.2, 5.0, 0.05),
    P("thresh", "Cut below", "int", 10, 0, 255, 1),
    *OUT_PARAMS,
])
def log_edges(img, sigma, ksize, gain, thresh, output, invert):
    g = cv2.GaussianBlur(to_gray(img).astype(np.float32), (0, 0), sigma)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=int(ksize)))
    return finish_edges(img, scale_response(lap, gain, thresh), output, invert)


@register("dog", "Edges", "Difference of Gaussians", "Band-pass edge detector, soft and clean.", [
    P("s1", "Sigma 1", "float", 1.0, 0.3, 5.0, 0.1),
    P("s2", "Sigma 2", "float", 2.0, 0.5, 10.0, 0.1),
    P("gain", "Gain", "float", 1.0, 0.2, 5.0, 0.05),
    P("thresh", "Cut below", "int", 5, 0, 255, 1),
    *OUT_PARAMS,
])
def dog(img, s1, s2, gain, thresh, output, invert):
    g = to_gray(img).astype(np.float32)
    d = np.abs(cv2.GaussianBlur(g, (0, 0), s1) - cv2.GaussianBlur(g, (0, 0), max(s2, s1 + 0.1)))
    return finish_edges(img, scale_response(d, gain, thresh), output, invert)


@register("morph_edge", "Edges", "Morphological gradient", "Thick outlines (dilate - erode).", [
    P("ksize", "Kernel size", "int", 3, 1, 15, 2),
    P("gain", "Gain", "float", 1.0, 0.2, 5.0, 0.05),
    P("thresh", "Cut below", "int", 10, 0, 255, 1),
    *OUT_PARAMS,
])
def morph_edge(img, ksize, gain, thresh, output, invert):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(ksize), odd(ksize)))
    m = cv2.morphologyEx(to_gray(img), cv2.MORPH_GRADIENT, k).astype(np.float32)
    return finish_edges(img, scale_response(m, gain, thresh), output, invert)


@register("contours", "Edges", "Contours", "Find and draw object contours (optionally simplified polygons).", [
    P("low", "Canny low", "int", 50, 0, 500, 1),
    P("high", "Canny high", "int", 150, 0, 500, 1),
    P("blur", "Gaussian blur (odd)", "int", 5, 1, 21, 2),
    P("min_length", "Min contour length (px)", "int", 40, 0, 2000, 5),
    P("approx", "Polygon simplify", "float", 0.0, 0.0, 0.05, 0.001),
    P("thickness", "Line thickness", "int", 2, 1, 10, 1),
    P("background", "Background", "choice", "original", choices=("original", "black", "white")),
])
def contours(img, low, high, blur, min_length, approx, thickness, background):
    g = cv2.GaussianBlur(to_gray(img), (odd(blur), odd(blur)), 0)
    e = cv2.Canny(g, low, max(high, low + 1))
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    keep = []
    for c in cs:
        if cv2.arcLength(c, False) < min_length:
            continue
        if approx > 0:
            c = cv2.approxPolyDP(c, approx * cv2.arcLength(c, True), True)
        keep.append(c)
    if background == "black":
        canvas = np.zeros_like(ensure_bgr(img))
    elif background == "white":
        canvas = np.full_like(ensure_bgr(img), 255)
    else:
        canvas = ensure_bgr(img).copy()
    color = (0, 200, 0) if background == "original" else ((255, 255, 255) if background == "black" else (0, 0, 0))
    cv2.drawContours(canvas, keep, -1, color, thickness)
    return canvas


# ----------------------------------------------------------------------------
# ENHANCE
# ----------------------------------------------------------------------------


def _unsharp(img, amount, radius, threshold=0):
    f = img.astype(np.float32)
    d = f - cv2.GaussianBlur(f, (0, 0), radius)
    if threshold > 0:
        d = np.where(np.abs(d) < threshold, 0, d)
    return np.clip(f + amount * d, 0, 255)


@register("unsharp", "Enhance", "Unsharp mask (sharpen)", "Boost edge contrast / sharpness.", [
    P("amount", "Amount", "float", 1.2, 0.0, 5.0, 0.05),
    P("radius", "Radius", "float", 1.5, 0.3, 10.0, 0.1),
    P("threshold", "Threshold", "int", 0, 0, 50, 1),
])
def unsharp(img, amount, radius, threshold):
    return _unsharp(img, amount, radius, threshold)


@register("clahe", "Enhance", "CLAHE (local contrast)", "Adaptive histogram equalization on lightness.", [
    P("clip", "Clip limit", "float", 2.5, 0.5, 10.0, 0.1),
    P("tile", "Tile grid", "int", 8, 2, 32, 1),
])
def clahe(img, clip, tile):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clip, (int(tile), int(tile))).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@register("auto_levels", "Enhance", "Auto levels", "Stretch histogram between percentiles.", [
    P("low_pct", "Black point %", "float", 0.5, 0.0, 20.0, 0.1),
    P("high_pct", "White point %", "float", 99.5, 80.0, 100.0, 0.1),
    P("per_channel", "Per channel (also fixes color cast)", "bool", False),
])
def auto_levels(img, low_pct, high_pct, per_channel):
    f = img.astype(np.float32)
    if per_channel:
        out = np.empty_like(f)
        for c in range(3):
            lo, hi = np.percentile(f[..., c], (low_pct, high_pct))
            out[..., c] = (f[..., c] - lo) / max(hi - lo, 1) * 255
    else:
        lo, hi = np.percentile(to_gray(img), (low_pct, high_pct))
        out = (f - lo) / max(hi - lo, 1) * 255
    return np.clip(out, 0, 255)


@register("gamma", "Enhance", "Gamma", "Gamma > 1 brightens shadows.", [
    P("gamma", "Gamma", "float", 1.3, 0.2, 4.0, 0.05),
])
def gamma(img, gamma):
    lut = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


@register("brightness_contrast", "Enhance", "Brightness / Contrast", "", [
    P("contrast", "Contrast", "float", 1.2, 0.2, 3.0, 0.05),
    P("brightness", "Brightness", "int", 0, -100, 100, 1),
])
def brightness_contrast(img, contrast, brightness):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


@register("grayscale", "Enhance", "Grayscale", "", [])
def grayscale(img):
    return to_gray(img)


@register("invert", "Enhance", "Invert colors", "", [])
def invert(img):
    return 255 - img


# ----------------------------------------------------------------------------
# RESTORE
# ----------------------------------------------------------------------------


@register("denoise_nlm", "Restore", "Denoise (Non-local means)", "Strong noise / JPEG-artifact reduction. Slow on big images.", [
    P("h", "Luma strength", "float", 6.0, 1.0, 30.0, 0.5),
    P("hcolor", "Color strength", "float", 6.0, 1.0, 30.0, 0.5),
    P("template", "Template (odd)", "int", 7, 3, 15, 2),
    P("search", "Search (odd)", "int", 21, 7, 41, 2),
])
def denoise_nlm(img, h, hcolor, template, search):
    return cv2.fastNlMeansDenoisingColored(img, None, h, hcolor, odd(template), odd(search))


@register("bilateral", "Restore", "Bilateral (edge-preserving smooth)", "Smooths flat areas, keeps edges.", [
    P("d", "Diameter", "int", 9, 3, 25, 2),
    P("sigma_color", "Sigma color", "float", 50.0, 5.0, 200.0, 1),
    P("sigma_space", "Sigma space", "float", 50.0, 5.0, 200.0, 1),
])
def bilateral(img, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(img, int(d), sigma_color, sigma_space)


@register("median", "Restore", "Median filter", "Removes salt-and-pepper noise.", [
    P("ksize", "Kernel (odd)", "int", 3, 3, 15, 2),
])
def median(img, ksize):
    return cv2.medianBlur(img, odd(ksize))


def detect_block_size(img, max_b=48):
    """Return the pixel-block size of a pixelated/mosaic image, or 1 if none."""
    g = to_gray(img).astype(np.float32)
    dx = np.abs(np.diff(g, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(g, axis=0)).mean(axis=1)

    def ratio(d, c):
        best, tot, n = 0.0, d.sum(), len(d)
        for p in range(c):
            sel = d[p::c]
            if len(sel) < 3 or n - len(sel) <= 0:
                continue
            inner = (tot - sel.sum()) / (n - len(sel))
            best = max(best, float(sel.mean() / (inner + 1e-3)))
        return best

    best_c, best_r = 1, 0.0
    for c in range(2, max_b + 1):
        r = 0.5 * (ratio(dx, c) + ratio(dy, c))
        if r > best_r:
            best_c, best_r = c, r
    return best_c if best_r >= 2.5 else 1


@register("depixelate", "Restore", "De-pixelate / de-mosaic", "Detects blocky pixels (auto) and rebuilds a smooth image.", [
    P("block", "Block size (0 = auto)", "int", 0, 0, 64, 1),
    P("sharpen", "Sharpen after", "float", 0.4, 0.0, 3.0, 0.05),
])
def depixelate(img, block, sharpen):
    b = int(block) or detect_block_size(img)
    h, w = img.shape[:2]
    if b <= 1:  # no blocks found: just clean lightly
        out = cv2.bilateralFilter(img, 7, 40, 5)
    else:
        ys, xs = np.arange(b // 2, h, b), np.arange(b // 2, w, b)
        small = img[ys][:, xs]
        out = cv2.resize(small, (len(xs) * b, len(ys) * b), interpolation=cv2.INTER_CUBIC)[:h, :w]
    return _unsharp(out, sharpen, 1.2) if sharpen > 0 else out


@register("dechecker", "Restore", "Remove checker / halftone / moire (FFT)", "Finds periodic patterns in the spectrum and notches them out.", [
    P("sensitivity", "Sensitivity (lower = more removal)", "float", 8.0, 2.0, 40.0, 0.5),
    P("radius", "Notch radius", "int", 3, 1, 15, 1),
    P("protect", "Protect low freq (px)", "int", 6, 0, 60, 1),
    P("strength", "Strength", "float", 1.0, 0.1, 1.0, 0.05),
])
def dechecker(img, sensitivity, radius, protect, strength):
    h, w = img.shape[:2]
    H, W = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
    f = cv2.copyMakeBorder(img, 0, H - h, 0, W - w, cv2.BORDER_REFLECT)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    res = mag - cv2.GaussianBlur(mag, (0, 0), max(2.0, min(H, W) / 100))
    med = np.median(res)
    mad = np.median(np.abs(res - med)) * 1.4826 + 1e-6
    mask = (res - med) > sensitivity * mad
    yy, xx = np.ogrid[:H, :W]
    mask &= ((yy - H // 2) ** 2 + (xx - W // 2) ** 2) > protect ** 2
    if not mask.any():
        return img
    r = max(1, int(radius))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    notch = cv2.GaussianBlur(cv2.dilate(mask.astype(np.uint8), k).astype(np.float32), (0, 0), r / 2 + 0.5)
    filt = 1.0 - np.clip(notch, 0, 1) * strength
    out = np.empty((H, W, 3), np.float32)
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(f[..., c].astype(np.float32))) * filt
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    return np.clip(out[:h, :w], 0, 255)


def _psf(kind, size, angle):
    if kind == "motion":
        L = max(3, int(round(size)) | 1)
        k = np.zeros((L, L), np.float32)
        k[L // 2, :] = 1
        M = cv2.getRotationMatrix2D(((L - 1) / 2, (L - 1) / 2), angle, 1.0)
        k = cv2.warpAffine(k, M, (L, L))
    else:
        ks = odd(int(6 * size + 1))
        g = cv2.getGaussianKernel(ks, size)
        k = (g @ g.T).astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else k


@register("deblur", "Restore", "Deblur (Wiener deconvolution)", "Fix out-of-focus (gaussian) or motion blur, e.g. blurry plates.", [
    P("kind", "Blur type", "choice", "motion", choices=("motion", "gaussian")),
    P("size", "Blur size / length", "float", 5.0, 1.0, 40.0, 0.5),
    P("angle", "Motion angle", "float", 0.0, -90.0, 90.0, 1.0),
    P("noise", "Noise (higher = safer)", "float", 0.01, 0.0005, 0.3, 0.0005),
    P("method", "Method", "choice", "wiener", choices=("wiener", "richardson_lucy")),
    P("iters", "Iterations (Richardson-Lucy)", "int", 15, 1, 60, 1),
])
def deblur(img, kind, size, angle, noise, method="wiener", iters=15):
    k = _psf(kind, size, angle)
    if method == "richardson_lucy":
        return richardson_lucy(img, k, iters)
    kh, kw = k.shape
    pad = max(kh, kw)
    f = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    H, W = f.shape[:2]
    psf = np.zeros((H, W), np.float32)
    psf[:kh, :kw] = k
    psf = np.roll(psf, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    Hf = np.fft.fft2(psf)
    G = np.conj(Hf) / (np.abs(Hf) ** 2 + noise)
    out = np.empty(f.shape, np.float32)
    for c in range(3):
        out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(f[..., c].astype(np.float32)) * G))
    return np.clip(out[pad:-pad, pad:-pad], 0, 255)


def richardson_lucy(img, psf, iters):
    f = img.astype(np.float32) + 1e-3
    est, flip = f.copy(), np.ascontiguousarray(psf[::-1, ::-1])
    for _ in range(int(iters)):
        conv = cv2.filter2D(est, -1, flip, borderType=cv2.BORDER_REFLECT)
        est = est * cv2.filter2D(f / (conv + 1e-3), -1, psf, borderType=cv2.BORDER_REFLECT)
    return np.clip(est, 0, 255)


def estimate_motion_blur(img, max_len=40):
    """Estimate (length, angle_deg, found) of linear motion blur via the cepstrum."""
    g = to_gray(img).astype(np.float32)
    s = min(min(g.shape), 512)
    y0, x0 = (g.shape[0] - s) // 2, (g.shape[1] - s) // 2
    g = g[y0:y0 + s, x0:x0 + s]
    g = (g - g.mean()) * np.outer(np.hanning(s), np.hanning(s)).astype(np.float32)
    spec = np.log(np.abs(np.fft.fft2(g)) + 1e-2)
    cep = np.fft.fftshift(np.real(np.fft.ifft2(spec)))
    c = s // 2
    yy, xx = np.ogrid[:s, :s]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    mask = (r2 >= 9) & (r2 <= max_len ** 2)
    vals = cep[mask]
    idx = np.argmin(np.where(mask, cep, np.inf))
    dy, dx = (int(v) - c for v in np.unravel_index(idx, cep.shape))
    z = (vals.mean() - cep[dy + c, dx + c]) / (vals.std() + 1e-9)
    angle = math.degrees(math.atan2(-dy, dx))
    if angle > 90:
        angle -= 180
    if angle <= -90:
        angle += 180
    return math.hypot(dx, dy), angle, bool(z > 8.0)


@register("auto_deblur", "Restore", "Auto deblur (estimates the blur)", "Detects motion-blur length/angle from the image itself, then deconvolves. Falls back to Gaussian if no motion found.", [
    P("method", "Method", "choice", "wiener", choices=("wiener", "richardson_lucy")),
    P("noise", "Noise (Wiener)", "float", 0.02, 0.0005, 0.3, 0.0005),
    P("iters", "Iterations (Richardson-Lucy)", "int", 15, 1, 60, 1),
    P("max_len", "Max blur length", "int", 40, 5, 100, 1),
    P("fallback", "Gaussian size if no motion found (0 = skip)", "float", 0.0, 0.0, 6.0, 0.1),
])
def auto_deblur(img, method, noise, iters, max_len, fallback):
    L, ang, ok = estimate_motion_blur(img, max_len)
    if ok:
        kind, size, angle = "motion", L, ang
    elif fallback > 0:
        kind, size, angle = "gaussian", fallback, 0.0
    else:
        return img
    return deblur(img, kind, size, angle, noise, method, iters)


@register("upscale", "Restore", "Upscale (super-resolution style)", "Enlarge with progressive cubic + edge-preserving cleanup + sharpening.", [
    P("factor", "Scale factor", "float", 2.0, 0.25, 8.0, 0.25),
    P("method", "Method", "choice", "smart", choices=("smart", "lanczos", "cubic")),
    P("sharpen", "Final sharpen", "float", 0.6, 0.0, 3.0, 0.05),
])
def upscale(img, factor, method, sharpen):
    h, w = img.shape[:2]
    if h * w * factor * factor > 120e6:
        raise ValueError("Output would be too large (>120 MP)")
    if factor < 1:
        return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    if method == "lanczos":
        out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
    elif method == "cubic":
        out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    else:
        out, cur = img.astype(np.float32), 1.0
        while cur < factor - 1e-6:
            step = min(2.0, factor / cur)
            out = np.clip(cv2.resize(out, None, fx=step, fy=step, interpolation=cv2.INTER_CUBIC), 0, 255)
            out = cv2.bilateralFilter(out, 5, 25, 3)
            out = _unsharp(out, 0.5, 1.0)
            cur *= step
    return _unsharp(out, sharpen, 1.2) if sharpen > 0 else out


@register("ai_upscale", "Restore", "AI upscale (EDSR / ESPCN / FSRCNN / LapSRN)",
          "Neural super-resolution. Needs opencv-contrib-python and a .pb model whose name contains the scale, e.g. EDSR_x4.pb. Slow on big images.", [
    P("model", "Model file (.pb)", "str", "models/EDSR_x4.pb"),
])
def ai_upscale(img, model):
    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError("cv2.dnn_superres missing: pip uninstall opencv-python && pip install opencv-contrib-python")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"Model file not found: {model}")
    name = os.path.basename(model).lower()
    m = re.search(r"x(\d)", name)
    if not m:
        raise ValueError("Model file name must contain the scale, e.g. EDSR_x4.pb")
    algo = re.split(r"[_\-. ]", name)[0]
    if algo not in ("edsr", "espcn", "fsrcnn", "lapsrn"):
        raise ValueError("Model name must start with edsr / espcn / fsrcnn / lapsrn")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model)
    sr.setModel(algo, int(m.group(1)))
    return sr.upsample(img)


# ----------------------------------------------------------------------------
# THRESHOLD & MORPH
# ----------------------------------------------------------------------------


@register("binarize", "Threshold & Morph", "Binarize (OCR-friendly)", "Black/white conversion for plates & documents.", [
    P("method", "Method", "choice", "adaptive_gaussian", choices=("adaptive_gaussian", "adaptive_mean", "otsu")),
    P("block", "Block size (odd)", "int", 31, 3, 101, 2),
    P("c", "Constant C", "int", 10, -30, 50, 1),
    P("invert", "Invert", "bool", False),
])
def binarize(img, method, block, c, invert):
    g = to_gray(img)
    t = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "otsu":
        _, out = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, t + cv2.THRESH_OTSU)
    else:
        m = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "adaptive_gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
        out = cv2.adaptiveThreshold(g, 255, m, t, odd(block), c)
    return out


@register("morphology", "Threshold & Morph", "Morphology", "Erode / dilate / open / close / top-hat ...", [
    P("operation", "Operation", "choice", "close",
      choices=("erode", "dilate", "open", "close", "gradient", "tophat", "blackhat")),
    P("ksize", "Kernel size", "int", 3, 1, 31, 2),
    P("shape", "Shape", "choice", "rect", choices=("rect", "ellipse", "cross")),
    P("iters", "Iterations", "int", 1, 1, 10, 1),
])
def morphology(img, operation, ksize, shape, iters):
    ops = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN,
           "close": cv2.MORPH_CLOSE, "gradient": cv2.MORPH_GRADIENT,
           "tophat": cv2.MORPH_TOPHAT, "blackhat": cv2.MORPH_BLACKHAT}
    shp = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}[shape]
    k = cv2.getStructuringElement(shp, (odd(ksize), odd(ksize)))
    return cv2.morphologyEx(img, ops[operation], k, iterations=int(iters))


# ----------------------------------------------------------------------------
# GEOMETRY
# ----------------------------------------------------------------------------


@register("rotate", "Geometry", "Rotate (free angle)", "Positive = counter-clockwise.", [
    P("angle", "Angle", "float", 0.0, -180.0, 180.0, 0.1),
    P("expand", "Expand canvas", "bool", True),
])
def rotate(img, angle, expand):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    size = (w, h)
    if expand:
        c, s = abs(M[0, 0]), abs(M[0, 1])
        size = (int(h * s + w * c), int(h * c + w * s))
        M[0, 2] += size[0] / 2 - w / 2
        M[1, 2] += size[1] / 2 - h / 2
    return cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


@register("flip", "Geometry", "Flip / rotate 90", "", [
    P("mode", "Mode", "choice", "horizontal",
      choices=("horizontal", "vertical", "both", "rot90_cw", "rot90_ccw", "rot180")),
])
def flip(img, mode):
    return {
        "horizontal": lambda a: cv2.flip(a, 1), "vertical": lambda a: cv2.flip(a, 0),
        "both": lambda a: cv2.flip(a, -1), "rot180": lambda a: cv2.rotate(a, cv2.ROTATE_180),
        "rot90_cw": lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE),
        "rot90_ccw": lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }[mode](img)


@register("deskew", "Geometry", "Auto deskew (straighten text / lines)", "Finds the dominant near-horizontal line angle with Hough and rotates it level.", [
    P("max_angle", "Max angle to correct", "float", 15.0, 1.0, 45.0, 1.0),
    P("expand", "Expand canvas", "bool", True),
])
def deskew(img, max_angle, expand):
    g = to_gray(img)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(e, 1, np.pi / 720, threshold=60, minLineLength=max(20, min(g.shape) // 6), maxLineGap=10)
    if lines is None:
        return img
    angs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = a - 180 if a > 90 else (a + 180 if a < -90 else a)
        if abs(a) <= max_angle:
            angs.append(a)
    return rotate(img, float(np.median(angs)), expand) if angs else img


# ----------------------------------------------------------------------------
# PERSPECTIVE
# ----------------------------------------------------------------------------

INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
          "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}

ASPECT_PRESETS = {
    "Auto (from quad)": 0.0,
    "Iran plate 4:1": 4.0,
    "EU plate 4.73:1": 520 / 110,
    "US plate 2:1": 2.0,
    "A4 portrait": 210 / 297,
    "A4 landscape": 297 / 210,
    "ID card 1.586:1": 1.586,
    "Square 1:1": 1.0,
}


def order_points(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def warp_quad(img, quad, aspect=0.0, scale=1.0, interp="cubic"):
    tl, tr, br, bl = order_points(quad)
    W = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    H = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if aspect > 0:
        H = W / aspect
    W, H = int(round(W * scale)), int(round(H * scale))
    W, H = max(8, min(W, 10000)), max(8, min(H, 10000))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=INTERP[interp], borderMode=cv2.BORDER_REPLICATE)


def _approx_quad(cnt):
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for e in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        ap = cv2.approxPolyDP(hull, e * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def find_quads(img, plate=False, min_area=0.004, max_area=0.95, limit=8):
    """Return candidate quadrilaterals (4x2 float arrays in image coords), best first."""
    h, w = img.shape[:2]
    sc = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img
    g = cv2.bilateralFilter(to_gray(small), 9, 60, 60)
    v = float(np.median(g))
    maps = [cv2.Canny(g, int(max(0, 0.67 * v)), int(min(255, 1.33 * v))),
            cv2.Canny(g, 30, 100)]
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    maps.append(th)
    area_img = small.shape[0] * small.shape[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    found = []
    for em in maps:
        em = cv2.morphologyEx(cv2.dilate(em, k), cv2.MORPH_CLOSE, k)
        cs, _ = cv2.findContours(em, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cs, key=cv2.contourArea, reverse=True)[:40]:
            ca = cv2.contourArea(c)
            if not (min_area <= ca / area_img <= max_area):
                continue
            q = _approx_quad(c)
            qa = cv2.contourArea(q)
            if qa < 1 or ca / qa < 0.6:
                continue
            o = order_points(q)
            wq = (np.linalg.norm(o[1] - o[0]) + np.linalg.norm(o[2] - o[3])) / 2
            hq = (np.linalg.norm(o[3] - o[0]) + np.linalg.norm(o[2] - o[1])) / 2
            if wq < 5 or hq < 5:
                continue
            if plate and not (1.8 <= wq / hq <= 8.0):
                continue
            found.append((qa * min(1.0, ca / qa), q / sc))
    found.sort(key=lambda t: t[0], reverse=True)
    diag = math.hypot(h, w)
    out = []
    for score, q in found:
        c = q.mean(axis=0)
        if any(np.linalg.norm(c - o.mean(axis=0)) < 0.04 * diag
               and abs(cv2.contourArea(q) / max(cv2.contourArea(o), 1) - 1) < 0.2 for o in out):
            continue
        out.append(q)
        if len(out) >= limit:
            break
    return out


@register("perspective", "Perspective", "Perspective correction", "quad=auto or x1|y1|...|x4|y4", [
    P("quad", "Quad", "str", "auto"),
    P("plate", "Plate aspect filter (auto)", "bool", False),
    P("candidate", "Auto candidate #", "int", 0, 0, 7, 1),
    P("aspect", "Aspect w/h (0 = auto)", "float", 0.0, 0.0, 8.0, 0.01),
    P("scale", "Output scale", "float", 1.0, 0.25, 4.0, 0.05),
    P("interp", "Interpolation", "choice", "cubic", choices=tuple(INTERP)),
])
def perspective(img, quad, plate, candidate, aspect, scale, interp):
    if str(quad).strip().lower() == "auto":
        qs = find_quads(img, plate=bool(plate))
        if not qs:
            raise ValueError("No quadrilateral found - give quad=x1|y1|... manually")
        q = qs[min(int(candidate), len(qs) - 1)]
    else:
        vals = [float(v) for v in str(quad).replace("|", ",").split(",") if v.strip()]
        if len(vals) != 8:
            raise ValueError("quad needs 8 numbers")
        q = np.array(vals, np.float32).reshape(4, 2)
    return warp_quad(img, q, aspect, scale, interp)


@register("plate_enhance", "Plate", "Plate / sign - one click", "Auto-detect the plate, straighten it, then denoise, auto-deblur, normalise height, local contrast and sharpen.", [
    P("detect", "Auto-detect & straighten", "bool", True),
    P("aspect", "Aspect w/h (0 = auto)", "float", 4.0, 0.0, 8.0, 0.01),
    P("height", "Output height (px)", "int", 220, 60, 800, 10),
    P("denoise", "Denoise strength", "float", 4.0, 0.0, 20.0, 0.5),
    P("fix_blur", "Auto deblur", "bool", True),
    P("contrast", "Local contrast (CLAHE)", "bool", True),
    P("sharpen", "Sharpen", "float", 1.0, 0.0, 3.0, 0.05),
    P("bw", "Black/white output", "bool", False),
])
def plate_enhance(img, detect, aspect, height, denoise, fix_blur, contrast, sharpen, bw):
    out = img
    if detect:
        qs = find_quads(img, plate=True)
        if qs:
            out = warp_quad(img, qs[0], aspect, 1.0, "cubic")
    s = height / out.shape[0]
    out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    if denoise > 0:
        out = cv2.fastNlMeansDenoisingColored(out, None, denoise, denoise, 7, 21)
    if fix_blur:
        out = ensure_bgr(auto_deblur(out, "wiener", 0.02, 15, 30, 0.0))
    if contrast:
        out = clahe(out, 2.0, 4)
    if sharpen > 0:
        out = ensure_bgr(_unsharp(out, sharpen, 1.2))
    return ensure_bgr(binarize(out, "adaptive_gaussian", 31, 10, False)) if bw else out


# ----------------------------------------------------------------------------
# OCR (optional)
# ----------------------------------------------------------------------------

OCR_TESS_LANG = "eng"        # e.g. "fas+eng" if the Persian traineddata is installed
OCR_EASY_LANGS = ["en"]      # e.g. ["fa", "en"]


def ocr_text(img):
    g = to_gray(img)
    tess_err = None
    try:
        import pytesseract
        return pytesseract.image_to_string(g, lang=OCR_TESS_LANG, config="--psm 7").strip()
    except ImportError:
        pass
    except Exception as e:  # Tesseract binary missing, bad language ...
        tess_err = e
    try:
        import easyocr
    except ImportError:
        raise RuntimeError(str(tess_err) if tess_err else
                           "OCR needs pytesseract (+Tesseract) or easyocr:  pip install easyocr")
    reader = easyocr.Reader(OCR_EASY_LANGS, gpu=False, verbose=False)
    return " ".join(reader.readtext(ensure_bgr(img), detail=0)).strip()


# ----------------------------------------------------------------------------
# Pipeline (CLI + recipe export)
# ----------------------------------------------------------------------------


def coerce(op: Op, kv: dict):
    vals, pm = op.defaults(), {p.key: p for p in op.params}
    for k, raw in kv.items():
        if k not in pm:
            raise KeyError(f"{op.name}: unknown parameter '{k}'")
        kind = pm[k].kind
        if kind == "int":
            vals[k] = int(float(raw))
        elif kind == "float":
            vals[k] = float(raw)
        elif kind == "bool":
            vals[k] = str(raw).lower() in ("1", "true", "yes", "y", "on")
        else:
            vals[k] = str(raw)
    return vals


def parse_pipeline(text):
    steps = []
    for part in (p.strip() for p in text.split(";")):
        if not part:
            continue
        name, _, rest = part.partition(":")
        kv = {}
        for item in filter(None, (i.strip() for i in rest.split(","))):
            k, _, v = item.partition("=")
            kv[k.strip()] = v.strip()
        if name.strip() not in OPS:
            raise KeyError(f"Unknown operation '{name}'. Use --list.")
        steps.append((name.strip(), kv))
    return steps


def run_pipeline(img, steps):
    for name, kv in steps:
        op = OPS[name]
        img = ensure_bgr(op.func(img.copy(), **coerce(op, kv))) if name != "grayscale" else ensure_bgr(op.func(img))
    return img


def recipe_string(recipe):
    def fmt(v):
        return ("1" if v else "0") if isinstance(v, bool) else (f"{v:g}" if isinstance(v, float) else str(v))
    return "; ".join(f"{n}:" + ",".join(f"{k}={fmt(v)}" for k, v in p.items()) for n, p in recipe)


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def save_preset(path, steps):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "steps": [[n, p] for n, p in steps]}, f, indent=1, default=str)


def load_preset(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    steps = []
    for n, kv in d["steps"]:
        if n not in OPS:
            raise KeyError(f"Preset uses unknown operation '{n}'")
        steps.append((n, dict(kv)))
    return steps


def batch_steps(steps, auto_persp):
    """For batch runs a fixed perspective quad is meaningless; optionally re-detect per image."""
    out = []
    for n, kv in steps:
        kv = dict(kv)
        if n == "perspective" and auto_persp:
            kv["plate"] = float(kv.get("aspect", 0) or 0) >= 1.8
            kv["quad"], kv["candidate"] = "auto", 0
        out.append((n, kv))
    return out


def list_images(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(IMG_EXTS) and not os.path.splitext(f)[0].endswith("_out"))


def process_file(src, dst, steps):
    imwrite(dst, run_pipeline(imread(src), steps))


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------


class OpPanel(ttk.Frame):
    """Generic tab: choose an operation of a category, tweak parameters, preview/apply."""

    def __init__(self, master, app, ops):
        super().__init__(master)
        self.app, self.ops = app, ops
        self.op_var = tk.StringVar(value=ops[0].label)
        cb = ttk.Combobox(self, textvariable=self.op_var, values=[o.label for o in ops], state="readonly")
        cb.pack(fill="x", padx=6, pady=(8, 2))
        cb.bind("<<ComboboxSelected>>", lambda e: self.rebuild())
        self.desc = ttk.Label(self, wraplength=340, foreground="#777", justify="left")
        self.desc.pack(fill="x", padx=8)
        self.box = ttk.Frame(self)
        self.box.pack(fill="both", expand=True, padx=6, pady=4)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=6)
        self.live = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Live preview", variable=self.live).pack(side="left")
        ttk.Button(bar, text="Apply", command=lambda: self.go(True)).pack(side="right", padx=2)
        ttk.Button(bar, text="Preview", command=lambda: self.go(False)).pack(side="right", padx=2)
        self._after = None
        self.rebuild()

    @property
    def op(self):
        return next(o for o in self.ops if o.label == self.op_var.get())

    def rebuild(self):
        for w in self.box.winfo_children():
            w.destroy()
        op = self.op
        self.desc.config(text=op.desc)
        self.vars = {}
        if not op.params:
            ttk.Label(self.box, text="(no parameters)").pack(pady=10)
        for p in op.params:
            if p.kind in ("int", "float"):
                var = tk.IntVar(value=p.default) if p.kind == "int" else tk.DoubleVar(value=p.default)
                ttk.Label(self.box, text=p.label).pack(anchor="w")
                tk.Scale(self.box, from_=p.lo, to=p.hi, resolution=p.step, orient="horizontal",
                         variable=var, showvalue=True).pack(fill="x")
            elif p.kind == "bool":
                var = tk.BooleanVar(value=p.default)
                ttk.Checkbutton(self.box, text=p.label, variable=var, command=self.changed).pack(anchor="w", pady=2)
            elif p.kind == "choice":
                var = tk.StringVar(value=p.default)
                ttk.Label(self.box, text=p.label).pack(anchor="w")
                c = ttk.Combobox(self.box, textvariable=var, values=p.choices, state="readonly")
                c.pack(fill="x", pady=(0, 4))
                c.bind("<<ComboboxSelected>>", lambda e: self.changed())
            elif p.kind == "str":
                var = tk.StringVar(value=p.default)
                ttk.Label(self.box, text=p.label).pack(anchor="w")
                row = ttk.Frame(self.box)
                row.pack(fill="x", pady=(0, 4))
                ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
                if p.key == "model":
                    ttk.Button(row, text="...", width=3,
                               command=lambda v=var: v.set(filedialog.askopenfilename() or v.get())).pack(side="left", padx=2)
            else:
                continue
            if p.kind != "str":
                var.trace_add("write", lambda *a: self.changed())
            self.vars[p.key] = var

    def values(self):
        out = {}
        for p in self.op.params:
            v = self.vars[p.key].get()
            out[p.key] = int(v) if p.kind == "int" else (float(v) if p.kind == "float" else v)
        return out

    def changed(self):
        if self.live.get():
            if self._after:
                self.after_cancel(self._after)
            self._after = self.after(250, lambda: self.go(False))

    def go(self, commit):
        if self.app.current is None:
            return
        op, vals, src = self.op, self.values(), self.app.current
        self.app.run(lambda: op.func(src.copy(), **vals) if op.name != "grayscale" else op.func(src),
                     commit, op.label, (op.name, vals))


class PerspectivePanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ttk.Label(self, text="1) Tick 'Pick corners' and click the 4 corners (drag to adjust)\n"
                             "   or press Auto-detect.   2) Preview / Apply.",
                  wraplength=340, justify="left", foreground="#777").pack(fill="x", padx=8, pady=8)
        self.pick = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Pick corners on image", variable=self.pick).pack(anchor="w", padx=8)
        self.plate = tk.BooleanVar(value=False)
        ttk.Checkbutton(self, text="Auto-detect: prefer plate-like (wide) shapes",
                        variable=self.plate).pack(anchor="w", padx=8)
        row = ttk.Frame(self)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Button(row, text="Auto-detect", command=app.auto_quad).pack(side="left", padx=2)
        ttk.Button(row, text="Next candidate", command=app.next_quad).pack(side="left", padx=2)
        ttk.Button(row, text="Clear points", command=app.clear_pts).pack(side="left", padx=2)

        ttk.Label(self, text="Output shape").pack(anchor="w", padx=8, pady=(10, 0))
        self.preset = tk.StringVar(value="Auto (from quad)")
        cb = ttk.Combobox(self, textvariable=self.preset, values=list(ASPECT_PRESETS), state="readonly")
        cb.pack(fill="x", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.aspect.set(round(ASPECT_PRESETS[self.preset.get()], 3)))
        ttk.Label(self, text="Aspect w/h (0 = auto)").pack(anchor="w", padx=8)
        self.aspect = tk.DoubleVar(value=0.0)
        tk.Scale(self, from_=0, to=8, resolution=0.01, orient="horizontal", variable=self.aspect).pack(fill="x", padx=8)
        ttk.Label(self, text="Output scale").pack(anchor="w", padx=8)
        self.scale = tk.DoubleVar(value=1.0)
        tk.Scale(self, from_=0.25, to=4, resolution=0.05, orient="horizontal", variable=self.scale).pack(fill="x", padx=8)
        ttk.Label(self, text="Interpolation").pack(anchor="w", padx=8)
        self.interp = tk.StringVar(value="cubic")
        ttk.Combobox(self, textvariable=self.interp, values=tuple(INTERP), state="readonly").pack(fill="x", padx=8)
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=12)
        ttk.Button(bar, text="Apply", command=lambda: self.go(True)).pack(side="right", padx=2)
        ttk.Button(bar, text="Preview", command=lambda: self.go(False)).pack(side="right", padx=2)

    def go(self, commit):
        app = self.app
        if app.current is None:
            return
        if len(app.pts) != 4:
            messagebox.showinfo("Perspective", "Select exactly 4 corners first (or use Auto-detect).")
            return
        q = np.array(app.pts, np.float32)
        a, s, it, src = float(self.aspect.get()), float(self.scale.get()), self.interp.get(), app.current
        qs = "|".join(f"{v:.1f}" for pt in app.pts for v in pt)
        app.run(lambda: warp_quad(src, q, a, s, it), commit, "Perspective",
                ("perspective", {"quad": qs, "aspect": a, "scale": s, "interp": it}))


class PipelinePanel(ttk.Frame):
    """Shows applied steps; reorder/remove, rebuild from the original, presets, batch folder."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ttk.Label(self, text="Applied steps. Reorder / remove, then 'Rebuild from original'.\n"
                             "Presets and batch use these steps.",
                  wraplength=340, justify="left", foreground="#777").pack(fill="x", padx=8, pady=8)
        self.lb = tk.Listbox(self, height=14, exportselection=False)
        self.lb.pack(fill="both", expand=True, padx=8)
        r1 = ttk.Frame(self)
        r1.pack(fill="x", padx=6, pady=4)
        for t, c in (("Up", self.up), ("Down", self.down), ("Remove", self.remove)):
            ttk.Button(r1, text=t, command=c).pack(side="left", padx=2)
        ttk.Button(self, text="Rebuild from original", command=app.rebuild).pack(fill="x", padx=8, pady=2)
        r2 = ttk.Frame(self)
        r2.pack(fill="x", padx=6, pady=4)
        ttk.Button(r2, text="Save preset...", command=self.save).pack(side="left", padx=2)
        ttk.Button(r2, text="Load preset...", command=self.load).pack(side="left", padx=2)
        ttk.Separator(self).pack(fill="x", padx=8, pady=8)
        self.auto_persp = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Batch: re-detect perspective per image",
                        variable=self.auto_persp).pack(anchor="w", padx=8)
        ttk.Button(self, text="Batch process folder...", command=self.batch).pack(fill="x", padx=8, pady=6)
        self.refresh()

    def refresh(self):
        self.lb.delete(0, "end")
        for i, (n, p) in enumerate(self.app.recipe, 1):
            txt = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in p.items() if k != "quad")
            self.lb.insert("end", f"{i}. {n}  ({txt[:55]})")

    def _sel(self):
        s = self.lb.curselection()
        return s[0] if s else None

    def _move(self, i, j):
        r = self.app.recipe
        r[i], r[j] = r[j], r[i]
        self.refresh()
        self.lb.selection_set(j)

    def up(self):
        i = self._sel()
        if i:
            self._move(i, i - 1)

    def down(self):
        i = self._sel()
        if i is not None and i < len(self.app.recipe) - 1:
            self._move(i, i + 1)

    def remove(self):
        i = self._sel()
        if i is not None:
            del self.app.recipe[i]
            self.refresh()

    def save(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Preset", "*.json")])
        if p:
            save_preset(p, self.app.recipe)
            self.app.msg.set(f"Preset saved: {p}")

    def load(self):
        p = filedialog.askopenfilename(filetypes=[("Preset", "*.json")])
        if not p:
            return
        try:
            steps = load_preset(p)
        except Exception as e:
            messagebox.showerror("Preset", str(e))
            return
        self.app.recipe = steps
        self.refresh()
        self.app.rebuild()

    def batch(self):
        if not self.app.recipe:
            messagebox.showinfo("Batch", "Apply at least one step first (or load a preset).")
            return
        src = filedialog.askdirectory(title="Input folder")
        if not src:
            return
        dst = filedialog.askdirectory(title="Output folder")
        if dst:
            self.app.start_batch(src, dst, self.auto_persp.get())


class App(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title("Image Lab - Edges / Perspective / Restore")
        self.geometry("1355x730")
        self.original = self.current = self.preview = None
        self.history, self.redo_stack, self.recipe = [], [], []
        self.zoom, self.off = 1.0, [0.0, 0.0]
        self.pts, self.cands, self.cand_i = [], [], 0
        self.compare = False
        self.split_on = tk.BooleanVar(value=False)
        self.split_pos = 0.5
        self._drag_split = False
        self.q, self.aux_q = queue.Queue(), queue.Queue()
        self.req = 0
        self.committing = False
        self._cache, self._orig_r = {}, None
        self._photo = None
        self._drag_idx = None
        self._pan = None
        self._batch = None
        self._build()
        self.after(40, self._poll)
        if path:
            self.after(100, lambda: self.load(path))

    # ---- UI ------------------------------------------------------------
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x")
        for text, cmd in [("Open", self.open), ("Save", self.save), ("Undo", self.undo), ("Redo", self.redo),
                          ("Reset", self.reset), ("Fit", self.fit), ("1:1", self.one_to_one),
                          ("OCR", self.do_ocr), ("Copy CLI recipe", self.copy_recipe)]:
            ttk.Button(top, text=text, command=cmd).pack(side="left", padx=2, pady=4)
        ttk.Checkbutton(top, text="Split view (original | edited)", variable=self.split_on,
                        command=self.render).pack(side="left", padx=10)
        b = ttk.Button(top, text="Hold: show original")
        b.pack(side="left", padx=4)
        b.bind("<ButtonPress-1>", lambda e: self.set_compare(True))
        b.bind("<ButtonRelease-1>", lambda e: self.set_compare(False))

        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(pw, bg="#282828", highlightthickness=0, cursor="crosshair")
        pw.add(self.canvas, weight=1)
        nb = ttk.Notebook(pw, width=390)
        pw.add(nb)
        self.panels = {}
        for cat in ("Edges", "Plate", "Restore", "Enhance", "Threshold & Morph", "Geometry"):
            self.panels[cat] = OpPanel(nb, self, [o for o in OPS.values() if o.category == cat])
        self.persp = PerspectivePanel(nb, self)
        self.pipeline = PipelinePanel(nb, self)
        nb.add(self.panels["Edges"], text="Edges")
        nb.add(self.persp, text="Perspective")
        nb.add(self.panels["Plate"], text="Plate")
        for cat in ("Restore", "Enhance", "Threshold & Morph", "Geometry"):
            nb.add(self.panels[cat], text=cat)
        nb.add(self.pipeline, text="Pipeline")

        st = ttk.Frame(self)
        st.pack(fill="x")
        self.msg = tk.StringVar(value="Open an image (Ctrl+O)")
        self.info = tk.StringVar()
        ttk.Label(st, textvariable=self.msg).pack(side="left", padx=6)
        ttk.Label(st, textvariable=self.info).pack(side="right", padx=6)

        c = self.canvas
        c.bind("<Configure>", lambda e: self.render())
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._drag)
        c.bind("<ButtonRelease-1>", self._release)
        for n in (2, 3):
            c.bind(f"<ButtonPress-{n}>", self._pan_start)
            c.bind(f"<B{n}-Motion>", self._pan_move)
        c.bind("<MouseWheel>", lambda e: self._zoom_at(e.x, e.y, 1.15 if e.delta > 0 else 1 / 1.15))
        c.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.15))
        c.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.15))
        c.bind("<Motion>", self._motion)
        self.bind("<Control-o>", lambda e: self.open())
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())

    # ---- file ----------------------------------------------------------
    def open(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All", "*.*")])
        if p:
            self.load(p)

    def load(self, path):
        try:
            img = imread(path)
        except Exception as e:
            messagebox.showerror("Open", str(e))
            return
        self.original, self.current, self.preview = img, img.copy(), None
        self.history, self.redo_stack, self.recipe = [], [], []
        self.pts, self.cands = [], []
        self._cache.clear()
        self._orig_r = None
        self.pipeline.refresh()
        self.title(f"Image Lab - {os.path.basename(path)}")
        self.update_idletasks()
        self.fit()
        self.msg.set(f"Loaded {img.shape[1]}x{img.shape[0]}")

    def save(self):
        img = self.preview if self.preview is not None else self.current
        if img is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tif")])
        if p:
            imwrite(p, img)
            self.msg.set(f"Saved {p}")

    def copy_recipe(self):
        s = recipe_string(self.recipe)
        self.clipboard_clear()
        self.clipboard_append(s)
        self.msg.set("CLI pipeline copied: " + (s[:90] + "..." if len(s) > 90 else s))

    # ---- OCR / batch (background work, results via aux_q) -----------------
    def do_ocr(self):
        img = self.preview if self.preview is not None else self.current
        if img is None:
            return
        self.msg.set("Running OCR ...")

        def work():
            try:
                self.aux_q.put(("ocr", ocr_text(img), None))
            except Exception as e:  # noqa
                self.aux_q.put(("ocr", None, str(e)))
        threading.Thread(target=work, daemon=True).start()

    def start_batch(self, indir, outdir, auto_persp):
        files = list_images(indir)
        if not files:
            messagebox.showinfo("Batch", "No images found in that folder.")
            return
        os.makedirs(outdir, exist_ok=True)
        steps = batch_steps(list(self.recipe), auto_persp)
        win = tk.Toplevel(self)
        win.title("Batch processing")
        pb = ttk.Progressbar(win, length=380, maximum=len(files))
        pb.pack(padx=14, pady=(14, 6))
        lbl = ttk.Label(win, text="")
        lbl.pack(padx=14, pady=(0, 14))
        self._batch = (win, pb, lbl)

        def work():
            ok = fail = 0
            for i, f in enumerate(files, 1):
                err = None
                try:
                    process_file(os.path.join(indir, f), os.path.join(outdir, os.path.splitext(f)[0] + "_out.png"), steps)
                    ok += 1
                except Exception as e:  # noqa
                    err, fail = str(e), fail + 1
                self.aux_q.put(("batch", i, len(files), f, err))
            self.aux_q.put(("batch_done", ok, fail))
        threading.Thread(target=work, daemon=True).start()

    # ---- processing ------------------------------------------------------
    def run(self, fn, commit, label, recipe=None, replace_recipe=None):
        if self.current is None:
            return
        if commit:
            if self.committing:
                return
            self.committing = True
        self.req += 1
        rid = self.req
        self.msg.set(f"Processing: {label} ...")

        def work():
            t = time.time()
            try:
                self.q.put((rid, commit, label, recipe, ensure_bgr(fn()), None, time.time() - t, replace_recipe))
            except Exception as e:  # noqa
                self.q.put((rid, commit, label, recipe, None, e, 0, replace_recipe))
        threading.Thread(target=work, daemon=True).start()

    def rebuild(self):
        if self.original is None:
            return
        steps = list(self.recipe)
        self.run(lambda: run_pipeline(self.original, steps), True, "Rebuild pipeline", None, steps)

    def _poll(self):
        try:
            while True:
                rid, commit, label, recipe, res, err, dt, replace = self.q.get_nowait()
                if commit:
                    self.committing = False
                if err is not None:
                    self.msg.set(f"Error in {label}: {err}")
                    if commit:
                        messagebox.showerror(label, str(err))
                elif commit:
                    self.history.append((self.current, list(self.recipe)))
                    self.history = self.history[-20:]
                    self.redo_stack.clear()
                    if res.shape[:2] != self.current.shape[:2]:
                        self.pts, self.cands = [], []
                        self.current = res
                        self.fit()
                    self.current, self.preview = res, None
                    if replace is not None:
                        self.recipe = list(replace)
                    elif recipe:
                        self.recipe.append(recipe)
                    self.pipeline.refresh()
                    self.msg.set(f"Applied {label} ({dt:.2f}s) -> {res.shape[1]}x{res.shape[0]}")
                    self.render()
                elif rid == self.req:
                    self.preview = res
                    self.msg.set(f"Preview {label} ({dt:.2f}s) - press Apply to keep")
                    self.render()
        except queue.Empty:
            pass
        try:
            while True:
                item = self.aux_q.get_nowait()
                if item[0] == "ocr":
                    _, text, err = item
                    if err:
                        self.msg.set("OCR failed")
                        messagebox.showerror("OCR", err)
                    else:
                        self.clipboard_clear()
                        self.clipboard_append(text)
                        self.msg.set(f"OCR: {text}  (copied)")
                        messagebox.showinfo("OCR result", text or "(nothing recognised)")
                elif item[0] == "batch" and self._batch:
                    _, i, n, f, err = item
                    self._batch[1]["value"] = i
                    self._batch[2].config(text=f"{i}/{n}  {f}" + (f"   ERROR: {err}" if err else ""))
                elif item[0] == "batch_done":
                    _, ok, fail = item
                    if self._batch:
                        self._batch[0].destroy()
                        self._batch = None
                    messagebox.showinfo("Batch", f"Done: {ok} ok, {fail} failed.")
        except queue.Empty:
            pass
        self.after(40, self._poll)

    def undo(self):
        if self.history:
            self.redo_stack.append((self.current, list(self.recipe)))
            self.current, self.recipe = self.history.pop()
            self.preview, self.pts = None, []
            self.pipeline.refresh()
            self.render()

    def redo(self):
        if self.redo_stack:
            self.history.append((self.current, list(self.recipe)))
            self.current, self.recipe = self.redo_stack.pop()
            self.preview = None
            self.pipeline.refresh()
            self.render()

    def reset(self):
        if self.original is not None:
            self.history.append((self.current, list(self.recipe)))
            self.current, self.recipe, self.preview = self.original.copy(), [], None
            self.pts = []
            self.pipeline.refresh()
            self.fit()

    def set_compare(self, v):
        self.compare = v
        self.render()

    # ---- perspective points ---------------------------------------------
    def auto_quad(self):
        if self.current is None:
            return
        self.cands = find_quads(self.current, plate=self.persp.plate.get())
        self.cand_i = 0
        if not self.cands:
            self.msg.set("No quadrilateral found - pick corners manually")
            return
        self.pts = [list(map(float, p)) for p in order_points(self.cands[0])]
        self.msg.set(f"{len(self.cands)} candidate(s) found - 'Next candidate' cycles")
        self.render()

    def next_quad(self):
        if not self.cands:
            return self.auto_quad()
        self.cand_i = (self.cand_i + 1) % len(self.cands)
        self.pts = [list(map(float, p)) for p in order_points(self.cands[self.cand_i])]
        self.msg.set(f"Candidate {self.cand_i + 1}/{len(self.cands)}")
        self.render()

    def clear_pts(self):
        self.pts, self.cands = [], []
        self.render()

    # ---- view ------------------------------------------------------------
    def shown(self):
        if self.compare:
            return self.original
        return self.preview if self.preview is not None else self.current

    def fit(self):
        if self.current is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        h, w = self.current.shape[:2]
        self.zoom = min(cw / w, ch / h) * 0.97
        self.off = [(cw - w * self.zoom) / 2, (ch - h * self.zoom) / 2]
        self.render()

    def one_to_one(self):
        if self.current is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        h, w = self.current.shape[:2]
        self.zoom = 1.0
        self.off = [(cw - w) / 2, (ch - h) / 2]
        self.render()

    def _zoom_at(self, mx, my, f):
        z = min(max(self.zoom * f, 0.02), 40.0)
        f = z / self.zoom
        self.off = [mx - (mx - self.off[0]) * f, my - (my - self.off[1]) * f]
        self.zoom = z
        self.render()

    def _to_img(self, x, y):
        return [(x - self.off[0]) / self.zoom, (y - self.off[1]) / self.zoom]

    def _to_cv(self, p):
        return p[0] * self.zoom + self.off[0], p[1] * self.zoom + self.off[1]

    def _canvas_img(self, img, cw, ch):
        """Draw `img` (zoomed/panned) onto a cw x ch BGR array."""
        z = self.zoom
        if z < 1:
            ent = self._cache.get(id(img))
            if ent is None or ent[0] is not img or ent[1] != z:
                if len(self._cache) > 6:
                    self._cache.clear()
                ent = (img, z, cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_AREA))
                self._cache[id(img)] = ent
            src, M, fl = ent[2], np.float32([[1, 0, self.off[0]], [0, 1, self.off[1]]]), cv2.INTER_LINEAR
        else:
            src, M = img, np.float32([[z, 0, self.off[0]], [0, z, self.off[1]]])
            fl = cv2.INTER_NEAREST if z >= 4 else cv2.INTER_LINEAR
        return cv2.warpAffine(src, M, (cw, ch), flags=fl, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

    def render(self):
        img = self.shown()
        c = self.canvas
        cw, ch = c.winfo_width(), c.winfo_height()
        if img is None or cw < 2 or ch < 2:
            return
        out = self._canvas_img(img, cw, ch)
        split = self.split_on.get() and not self.compare and self.original is not None
        sx = int(self.split_pos * cw)
        if split:
            orig = self.original
            if orig.shape[:2] != img.shape[:2]:  # edited image has another size: align original to it
                if self._orig_r is None or self._orig_r[0] != img.shape[:2] or self._orig_r[2] is not orig:
                    self._orig_r = (img.shape[:2],
                                    cv2.resize(orig, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC), orig)
                orig = self._orig_r[1]
            out[:, :sx] = self._canvas_img(orig, cw, ch)[:, :sx]
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        self._photo = tk.PhotoImage(width=cw, height=ch, data=f"P6 {cw} {ch} 255 ".encode() + rgb.tobytes(), format="PPM")
        c.delete("all")
        c.create_image(0, 0, image=self._photo, anchor="nw")
        if split:
            c.create_line(sx, 0, sx, ch, fill="white", width=2)
            c.create_oval(sx - 14, ch // 2 - 14, sx + 14, ch // 2 + 14, fill="#222", outline="white", width=2)
            c.create_text(sx, ch // 2, text="<>", fill="white", font=("Arial", 11, "bold"))
            for x, anchor, t in ((10, "nw", "Original"), (cw - 10, "ne", "Edited")):
                c.create_text(x + 1, 11, text=t, anchor=anchor, fill="black", font=("Arial", 12, "bold"))
                c.create_text(x, 10, text=t, anchor=anchor, fill="white", font=("Arial", 12, "bold"))
        if self.pts and not self.compare:
            cp = [self._to_cv(p) for p in self.pts]
            if len(cp) == 4:
                o = order_points(self.pts)
                c.create_polygon([v for p in o for v in self._to_cv(p)], outline="#00ff66", fill="", width=2)
            elif len(cp) > 1:
                c.create_line([v for p in cp for v in p], fill="#00ff66", width=2)
            for i, (x, y) in enumerate(cp):
                c.create_oval(x - 7, y - 7, x + 7, y + 7, outline="yellow", fill="#e33", width=2)
                c.create_text(x + 14, y - 14, text=str(i + 1), fill="yellow", font=("Arial", 12, "bold"))

    # ---- mouse -----------------------------------------------------------
    def _near_split(self, x):
        return self.split_on.get() and self.current is not None and abs(x - self.split_pos * self.canvas.winfo_width()) < 14

    def _press(self, e):
        if self.current is None:
            return
        if self._near_split(e.x):
            self._drag_split = True
            return
        if self.persp.pick.get():
            for i, p in enumerate(self.pts):
                x, y = self._to_cv(p)
                if (x - e.x) ** 2 + (y - e.y) ** 2 < 15 ** 2:
                    self._drag_idx = i
                    return
            if len(self.pts) < 4:
                self.pts.append(self._to_img(e.x, e.y))
                self._drag_idx = len(self.pts) - 1
                self.render()
        else:
            self._pan_start(e)

    def _drag(self, e):
        if self._drag_split:
            self.split_pos = min(max(e.x / max(self.canvas.winfo_width(), 1), 0.0), 1.0)
            self.render()
        elif self._drag_idx is not None:
            h, w = self.current.shape[:2]
            x, y = self._to_img(e.x, e.y)
            self.pts[self._drag_idx] = [min(max(x, 0), w - 1), min(max(y, 0), h - 1)]
            self.render()
        elif not self.persp.pick.get():
            self._pan_move(e)

    def _release(self, e):
        self._drag_idx, self._drag_split = None, False

    def _pan_start(self, e):
        self._pan = (e.x, e.y, self.off[0], self.off[1])

    def _pan_move(self, e):
        if self._pan:
            self.off = [self._pan[2] + e.x - self._pan[0], self._pan[3] + e.y - self._pan[1]]
            self.render()

    def _motion(self, e):
        img = self.shown()
        if img is None:
            return
        self.canvas.config(cursor="sb_h_double_arrow" if self._near_split(e.x) else "crosshair")
        x, y = self._to_img(e.x, e.y)
        xi, yi = int(x), int(y)
        if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
            b, g, r = img[yi, xi]
            self.info.set(f"x={xi} y={yi}  RGB=({r},{g},{b})  zoom={self.zoom * 100:.0f}%  size={img.shape[1]}x{img.shape[0]}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Image Lab - edges, perspective, restoration")
    ap.add_argument("input", nargs="?", help="input image (or folder for batch)")
    ap.add_argument("-o", "--output", help="output image / folder (enables CLI mode)")
    ap.add_argument("-p", "--pipeline", default="", help='e.g. "perspective:quad=auto;unsharp:amount=1.5"')
    ap.add_argument("--preset", help="JSON preset saved from the GUI (instead of -p)")
    ap.add_argument("--auto-persp", action="store_true", help="batch: re-detect perspective for every image")
    ap.add_argument("--list", action="store_true", help="list operations and their parameters")
    a = ap.parse_args()

    if a.list:
        for op in OPS.values():
            print(f"\n[{op.category}] {op.name} - {op.label}")
            for p in op.params:
                extra = f" choices={list(p.choices)}" if p.choices else (f" range={p.lo}..{p.hi}" if p.kind in ("int", "float") else "")
                print(f"    {p.key} ({p.kind}, default={p.default}){extra}")
        return
    if a.input and a.output:
        steps = load_preset(a.preset) if a.preset else parse_pipeline(a.pipeline)
        if os.path.isdir(a.input):
            os.makedirs(a.output, exist_ok=True)
            steps = batch_steps(steps, a.auto_persp)
            files = list_images(a.input)
            for i, f in enumerate(files, 1):
                dst = os.path.join(a.output, os.path.splitext(f)[0] + "_out.png")
                try:
                    process_file(os.path.join(a.input, f), dst, steps)
                    print(f"[{i}/{len(files)}] {f} -> {dst}")
                except Exception as e:  # noqa
                    print(f"[{i}/{len(files)}] {f} FAILED: {e}")
            return
        img = run_pipeline(imread(a.input), steps)
        imwrite(a.output, img)
        print(f"Saved {a.output}  ({img.shape[1]}x{img.shape[0]})")
        return
    if not HAVE_TK:
        sys.exit("tkinter is not installed - use CLI mode (input -o output -p pipeline)")
    App(a.input).mainloop()

if __name__ == "__main__":
    main()