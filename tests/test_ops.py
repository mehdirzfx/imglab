import cv2
import numpy as np
import pytest

import imglab as L
from conftest import PLATE_QUAD, make_plate

NEEDS_EXTRAS = {"ai_upscale"}  # needs opencv-contrib and a downloaded model


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


@pytest.mark.parametrize("name", [n for n in L.OPS if n not in NEEDS_EXTRAS])
def test_every_operation_runs_with_defaults(name, scene):
    op = L.OPS[name]
    out = L.ensure_bgr(op.func(scene.copy(), **op.defaults()))
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3 and out.size > 0


def test_registry_is_consistent():
    for name, op in L.OPS.items():
        assert op.name == name and op.label and op.category
        for p in op.params:
            assert p.kind in ("int", "float", "bool", "choice", "str")
            if p.kind == "choice":
                assert str(p.default) in p.choices, (name, p.key)


def test_ai_upscale_reports_missing_model(scene, tmp_path):
    with pytest.raises((RuntimeError, FileNotFoundError)):
        L.ai_upscale(scene, str(tmp_path / "EDSR_x4.pb"))


def test_auto_detect_finds_the_plate(scene):
    quads = L.find_quads(scene, plate=True)
    assert quads, "no quadrilateral found"
    found = L.order_points(quads[0])
    truth = L.order_points(PLATE_QUAD)
    assert np.abs(found - truth).max() < 12


def test_perspective_output_has_requested_aspect(scene):
    out = L.perspective(scene, "auto", True, 0, 4.0, 1.0, "cubic")
    h, w = out.shape[:2]
    assert abs(w / h - 4.0) < 0.1


def test_manual_quad_matches_auto(scene):
    quad = "|".join(str(v) for v in PLATE_QUAD.ravel())
    out = L.perspective(scene, quad, False, 0, 4.0, 1.0, "cubic")
    assert out.shape[1] / out.shape[0] == pytest.approx(4.0, abs=0.1)
    # the straightened plate is mostly bright background with dark text
    assert out.mean() > 150


def test_perspective_rejects_bad_quad(scene):
    with pytest.raises(ValueError):
        L.perspective(scene, "1,2,3", False, 0, 0.0, 1.0, "cubic")


@pytest.mark.parametrize("block", [6, 10, 14])
def test_block_size_detection(block):
    plate = make_plate()
    small = cv2.resize(plate, (0, 0), fx=1 / block, fy=1 / block, interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, None, fx=block, fy=block, interpolation=cv2.INTER_NEAREST)
    assert L.detect_block_size(pix) == block


def test_sharp_image_is_not_blocky(text_img):
    assert L.detect_block_size(text_img) == 1


def test_dechecker_removes_pattern(text_img):
    yy, xx = np.indices(text_img.shape[:2])
    noisy = np.clip(text_img.astype(np.int16) + (((yy + xx) % 2) * 40 - 20)[..., None], 0, 255).astype(np.uint8)
    out = L.ensure_bgr(L.dechecker(noisy, 8.0, 3, 6, 1.0))
    assert mse(out, text_img) < 0.5 * mse(noisy, text_img)


@pytest.mark.parametrize("length,angle", [(5, 0), (9, 30), (15, 0), (15, 20), (11, -35)])
def test_motion_blur_estimate(length, angle, text_img):
    rng = np.random.default_rng(1)
    blurred = cv2.filter2D(text_img, -1, L._psf("motion", length, angle), borderType=cv2.BORDER_REFLECT)
    blurred = np.clip(blurred + rng.normal(0, 1, blurred.shape), 0, 255).astype(np.uint8)
    est_len, est_angle, found = L.estimate_motion_blur(blurred, 40)
    assert found
    assert abs(est_len - length) <= 2.5
    assert abs(est_angle - angle) <= 4


def test_sharp_images_are_not_flagged_as_blurred(text_img, scene):
    for img in (text_img, scene, make_plate()):
        assert L.estimate_motion_blur(img, 40)[2] is False


def test_blur_confidence_threshold_is_adjustable(text_img):
    blurred = cv2.filter2D(text_img, -1, L._psf("motion", 21, 60), borderType=cv2.BORDER_REFLECT)
    *_, conf = L.estimate_motion_blur(blurred, 40, return_score=True)
    assert L.estimate_motion_blur(blurred, 40, min_conf=conf - 1)[2] is True
    assert L.estimate_motion_blur(blurred, 40, min_conf=conf + 1)[2] is False


def test_auto_deblur_improves_a_blurred_image(text_img):
    blurred = cv2.filter2D(text_img, -1, L._psf("motion", 15, 20), borderType=cv2.BORDER_REFLECT)
    out = L.ensure_bgr(L.auto_deblur(blurred, "wiener", 0.02, 15, 40, 0.0))
    assert mse(out, text_img) < 0.7 * mse(blurred, text_img)


def test_auto_deblur_leaves_sharp_image_alone(text_img):
    assert np.array_equal(L.auto_deblur(text_img, "wiener", 0.02, 15, 40, 0.0), text_img)


@pytest.mark.parametrize("angle", [6.0, -4.0, 2.0, -9.0, 12.0])
def test_skew_estimate_matches_the_rotation(angle, text_img):
    rotated = L.rotate(text_img, angle, True)
    assert L.estimate_skew(rotated) == pytest.approx(-angle, abs=0.75)  # the correction undoes the rotation


def test_deskew_levels_the_image(text_img):
    rotated = L.rotate(text_img, 6.0, True)
    fixed = L.deskew(rotated, 15, True)
    assert abs(L.estimate_skew(fixed)) < 0.75


def test_skew_of_level_or_featureless_images(text_img):
    assert abs(L.estimate_skew(text_img)) < 0.3
    assert L.estimate_skew(np.full((200, 300, 3), 128, np.uint8)) is None


def test_upscale_and_downscale_sizes(scene):
    h, w = scene.shape[:2]
    assert L.upscale(scene, 2.0, "smart", 0.5).shape[:2] == (2 * h, 2 * w)
    assert L.upscale(scene, 0.5, "smart", 0.0).shape[:2] == (h // 2, w // 2)


def test_plate_enhance_normalises_height(scene):
    out = L.plate_enhance(scene, True, 4.0, 200, 2.0, True, True, 1.0, False)
    assert out.shape[0] == 200 and out.shape[1] == pytest.approx(800, abs=8)


def test_edge_outputs(scene):
    for out_mode in ("edges", "overlay", "keep_color"):
        out = L.canny(scene, True, 0.33, 50, 150, 5, "3", True, False, out_mode, False)
        assert out.shape == scene.shape
    edges = L.canny(scene, True, 0.33, 50, 150, 5, "3", True, False, "edges", False)
    assert 0 < (edges[..., 0] > 0).mean() < 0.2  # sparse edge map, not blank or all white


def test_binarize_gives_two_levels(scene):
    out = L.binarize(scene, "otsu", 31, 10, False)
    assert set(np.unique(out)) <= {0, 255}


def _mosaic(img, b):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // b, h // b), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w // b * b, h // b * b), interpolation=cv2.INTER_NEAREST)


def test_analyze_leaves_clean_images_alone(scene, text_img):
    for img in (scene, text_img):
        findings, steps = L.analyze(img)
        assert findings == [] and steps == []
        assert np.array_equal(L.auto_fix(img), img)


def test_analyze_finds_mosaic_blur_tilt_noise(scene, text_img):
    f, s = L.analyze(_mosaic(scene, 8))
    assert [n for n, _ in s][:1] == ["depixelate"] and s[0][1]["block"] == "8"
    k = np.zeros((9, 9), np.float32)
    k[4, :] = 1 / 9
    assert "auto_deblur" in [n for n, _ in L.analyze(cv2.filter2D(text_img, -1, k))[1]]
    assert "deskew" in [n for n, _ in L.analyze(L.rotate(text_img, 5, True))[1]]
    noisy = np.clip(scene + np.random.default_rng(0).normal(0, 15, scene.shape), 0, 255).astype(np.uint8)
    assert "denoise_nlm" in [n for n, _ in L.analyze(noisy)[1]]


def test_auto_fix_improves_a_mosaic(scene):
    bad = _mosaic(scene, 8)
    fixed = L.auto_fix(bad)
    assert fixed.shape == bad.shape and mse(fixed, scene[:bad.shape[0], :bad.shape[1]]) < mse(bad, scene[:bad.shape[0], :bad.shape[1]])


def test_make_comparison_has_both_halves(scene):
    after = L.rotate(scene, 90, True)
    out = L.make_comparison(scene, after, height=200)
    assert out.shape[0] == 200 and out.dtype == np.uint8
    assert out.shape[1] == round(scene.shape[1] * 200 / scene.shape[0]) + 12 + round(after.shape[1] * 200 / after.shape[0])
