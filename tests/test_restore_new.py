"""Tests for the restoration / histogram operations added in 1.2.0."""
import cv2
import numpy as np
import pytest

import imglab as L

NEW_OPS = ("dehaze", "low_light", "flatten_light", "deblock", "remove_spots", "levels", "equalize", "match_histogram")


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def jpeg(img, quality):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], 1)


@pytest.fixture(scope="module")
def plain_bg():
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(60, 140, (400, 600, 3)).astype(np.uint8), (0, 0), 25)


def test_new_operations_are_registered():
    for n in NEW_OPS:
        assert n in L.OPS
    assert "disk" in {p.key: p for p in L.OPS["deblur"].params}["kind"].choices


@pytest.mark.parametrize("name", NEW_OPS)
def test_new_operation_runs_through_the_cli_pipeline(name, scene):
    out = L.run_pipeline(scene, L.parse_pipeline(name))
    assert out.shape == scene.shape and out.dtype == np.uint8


# ---------------------------------------------------------------- dehaze / low light / lighting
def test_dehaze_restores_contrast_and_reduces_error(scene):
    clean = scene.astype(np.float32) / 255
    hazy = np.clip((clean * 0.5 + 0.9 * 0.5) * 255, 0, 255).astype(np.uint8)
    out = L.ensure_bgr(L.dehaze(hazy, 0.75, 15, 0.15))
    assert L.to_gray(out).std() > 1.5 * L.to_gray(hazy).std()
    assert mse(out, scene) < 0.2 * mse(hazy, scene)


def test_dehaze_strength_zero_is_nearly_identity(scene):
    out = L.ensure_bgr(L.dehaze(scene, 0.0, 15, 0.15))
    assert mse(out, scene) < 1.0


def test_low_light_brightens_dark_images(scene):
    dark = (scene * 0.2).astype(np.uint8)
    out = L.ensure_bgr(L.low_light(dark, 0.7, 0.0, 6.0))
    assert out.mean() > 3 * dark.mean()
    assert mse(out, scene) < 0.2 * mse(dark, scene)


def test_low_light_strength_zero_changes_nothing(scene):
    assert mse(L.ensure_bgr(L.low_light(scene, 0.0, 0.0, 6.0)), scene) < 1.0


def test_flatten_light_removes_a_shadow_gradient(text_img):
    h, w = text_img.shape[:2]
    shaded = (text_img * np.linspace(0.35, 1.0, w)[None, :, None]).astype(np.uint8)
    page = text_img.mean(axis=2) > 200  # paper pixels

    def drift(a):  # how much the paper brightness varies from column to column
        g = L.to_gray(a).astype(float)
        return np.std([g[:, i][page[:, i]].mean() for i in range(w) if page[:, i].any()])
    out = L.ensure_bgr(L.flatten_light(shaded, "document", 5.0, 1.0))
    assert drift(shaded) > 30 and drift(out) < 5
    assert mse(out, text_img) < 0.1 * mse(shaded, text_img)


def test_flatten_light_retinex_evens_out_a_photo(plain_bg):
    w = plain_bg.shape[1]
    shaded = (plain_bg * np.linspace(0.3, 1.0, w)[None, :, None]).astype(np.uint8)
    out = L.ensure_bgr(L.flatten_light(shaded, "retinex", 20.0, 1.0))
    col = lambda a: L.to_gray(a).astype(float).mean(axis=0)  # noqa: E731
    assert np.std(col(out)) < 0.4 * np.std(col(shaded))


# ---------------------------------------------------------------- deblock
def test_jpeg_grid_finds_the_grid_even_after_cropping(scene):
    j = jpeg(scene, 10)
    ox, oy, score = L.jpeg_grid(j)
    assert (ox, oy) == (0, 0) and score > 2
    ox, oy, _ = L.jpeg_grid(j[5:, 3:])  # 3 columns and 5 rows cropped: borders move to 5 and 3
    assert (ox, oy) == (5, 3)


def test_deblock_reduces_blockiness_without_hurting_quality(scene):
    j = jpeg(scene, 8)
    out = L.ensure_bgr(L.deblock(j, 8, 1.0, 40.0, 2, True))
    assert L.jpeg_grid(out)[2] < 0.9 * L.jpeg_grid(j)[2]
    assert mse(out, scene) <= mse(j, scene)


def test_deblock_keeps_sharp_edges(text_img):
    out = L.ensure_bgr(L.deblock(text_img, 8, 1.0, 40.0, 2, True))
    assert mse(out, text_img) < 1.0  # clean, sharp text must pass through (almost) untouched


def test_deblock_handles_tiny_images():
    tiny = np.full((5, 5, 3), 120, np.uint8)
    assert L.deblock(tiny, 8, 1.0, 40.0, 2, True).shape == tiny.shape


# ---------------------------------------------------------------- specks, defocus
def test_remove_spots_cleans_dust_and_leaves_clean_images_alone(plain_bg):
    rng = np.random.default_rng(3)
    dusty = plain_bg.copy()
    for _ in range(60):
        y, x = rng.integers(5, 390), rng.integers(5, 590)
        dusty[y:y + 2, x:x + 2] = 255 if rng.random() < 0.5 else 0
    out = L.ensure_bgr(L.remove_spots(dusty, "both", 30, 5, 3.0))
    assert mse(out, plain_bg) < 0.05 * mse(dusty, plain_bg)
    assert np.array_equal(L.remove_spots(plain_bg, "both", 30, 5, 3.0), plain_bg)


def test_remove_spots_ignores_texture(text_img):
    noisy = np.clip(text_img + np.random.default_rng(1).normal(0, 60, text_img.shape), 0, 255).astype(np.uint8)
    assert np.array_equal(L.remove_spots(noisy, "both", 5, 5, 3.0), noisy)  # >20 % flagged = texture, not dust


def test_disk_psf_is_normalised_and_round():
    k = L._psf("disk", 4, 0)
    assert k.sum() == pytest.approx(1.0) and k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert k[k.shape[0] // 2, 0] == 0 and k[k.shape[0] // 2, k.shape[1] // 2] > 0


def test_deblur_disk_undoes_defocus(text_img):
    blurred = cv2.filter2D(text_img, -1, L._psf("disk", 4, 0), borderType=cv2.BORDER_REFLECT)
    out = L.ensure_bgr(L.deblur(blurred, "disk", 4, 0, 0.005))
    assert mse(out, text_img) < 0.5 * mse(blurred, text_img)


# ---------------------------------------------------------------- levels / equalize / match
def test_levels_identity_and_stretch(scene):
    assert np.array_equal(L.levels(scene, "all", 0, 255, 1.0, 0, 255), scene)
    low = (scene * 0.5).astype(np.uint8)
    out = L.levels(low, "all", 0, int(low.max()), 1.0, 0, 255)
    assert out.max() == 255 and out.min() == 0 or out.max() >= 250


def test_levels_single_channel_only_touches_that_channel(scene):
    out = L.levels(scene, "red", 0, 255, 2.0, 0, 255)
    assert np.array_equal(out[..., :2], scene[..., :2]) and (out[..., 2] > scene[..., 2]).any()


def test_levels_gamma_and_degenerate_range(scene):
    assert L.levels(scene, "all", 0, 255, 2.0, 0, 255).mean() > scene.mean()
    L.levels(scene, "all", 200, 100, 1.0, 50, 10)  # inverted ranges are repaired, not an error


def test_equalize_flattens_the_histogram():
    rng = np.random.default_rng(0)
    narrow = rng.integers(100, 140, (200, 200, 3)).astype(np.uint8)
    out = L.equalize(narrow, "luminance", 1.0)
    lo, hi = np.percentile(L.to_gray(out), (1, 99))
    assert hi - lo > 200
    assert np.array_equal(L.equalize(narrow, "luminance", 0.0), narrow)


def test_match_histogram_adopts_reference_brightness(scene, tmp_path):
    ref = np.clip(scene.astype(np.int16) + 70, 0, 255).astype(np.uint8)
    path = tmp_path / "ref.png"
    L.imwrite(str(path), ref)
    out = L.match_histogram(scene, str(path), True, 1.0)
    assert abs(float(out.mean()) - float(ref.mean())) < 4
    assert np.array_equal(L.match_histogram(scene, "", True, 1.0), scene)
    assert abs(float(L.match_histogram(scene, str(path), False, 1.0).mean()) - float(ref.mean())) < 8


# ---------------------------------------------------------------- histogram helpers
def test_image_histograms_counts_every_pixel(scene):
    hist = L.image_histograms(scene)
    assert hist.shape == (4, 256)
    assert (hist.sum(axis=1) == scene.shape[0] * scene.shape[1]).all()
    b, g, r = (np.arange(256) * hist[i] for i in range(3))
    assert b.sum() / hist[0].sum() == pytest.approx(scene[..., 0].mean())


def test_image_histograms_samples_huge_images():
    big = np.zeros((4000, 4000, 3), np.uint8)
    big[:, 2000:] = 255
    hist = L.image_histograms(big, max_pixels=1_000_000)
    assert hist[3].sum() <= 1_100_000
    assert hist[3][0] == pytest.approx(hist[3][255], rel=0.01)  # sampling keeps the proportions


def test_image_histograms_accepts_gray_and_bgra():
    assert L.image_histograms(np.zeros((8, 8), np.uint8))[3][0] == 64
    assert L.image_histograms(np.zeros((8, 8, 4), np.uint8))[3][0] == 64


def test_histogram_stats():
    flat = np.zeros(256)
    flat[100] = 50
    s = L.histogram_stats(flat)
    assert s["mean"] == 100 and s["std"] == 0 and s["median"] == 100 and s["black"] == 0 and s["white"] == 0
    clip = np.zeros(256)
    clip[0], clip[255], clip[128] = 25, 25, 50
    s = L.histogram_stats(clip)
    assert s["black"] == 25 and s["white"] == 25 and s["median"] == 128
    assert L.histogram_stats(np.zeros(256))["mean"] == 0


# ---------------------------------------------------------------- auto fix knows about dark images
def test_analyze_suggests_low_light_for_dark_photos(scene):
    findings, steps = L.analyze((scene * 0.15).astype(np.uint8))
    assert any("dark" in f for f in findings)
    assert any(n == "low_light" for n, _ in steps) and not any(n == "auto_levels" for n, _ in steps)


def test_analyze_leaves_normal_images_alone(scene, text_img):
    for img in (scene, text_img):
        assert not any(n == "low_light" for n, _ in L.analyze(img)[1])
