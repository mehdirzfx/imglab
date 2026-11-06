import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# the quadrilateral of the plate in make_scene(), clockwise from top-left
PLATE_QUAD = np.float32([[170, 130], [430, 100], [440, 190], [175, 225]])


def make_plate():
    plate = np.full((110, 440, 3), 235, np.uint8)
    cv2.rectangle(plate, (0, 0), (439, 109), (20, 20, 20), 4)
    cv2.putText(plate, "12 B 345", (40, 80), cv2.FONT_HERSHEY_DUPLEX, 2.2, (10, 10, 10), 4)
    return plate


def make_scene():
    """600x400 grey background with a perspective-distorted plate."""
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(rng.integers(60, 140, (400, 600, 3)).astype(np.uint8), (0, 0), 25)
    src = np.float32([[0, 0], [439, 0], [439, 109], [0, 109]])
    M = cv2.getPerspectiveTransform(src, PLATE_QUAD)
    warped = cv2.warpPerspective(make_plate(), M, (600, 400))
    mask = cv2.warpPerspective(np.full((110, 440), 255, np.uint8), M, (600, 400))
    bg[mask > 0] = warped[mask > 0]
    return bg


def make_text():
    img = np.full((260, 640, 3), 235, np.uint8)
    for i, t in enumerate(["Image Lab 2026", "edges & blur 0123"]):
        cv2.putText(img, t, (25, 95 + i * 110), cv2.FONT_HERSHEY_DUPLEX, 1.9, (20, 20, 20), 3)
    return img


@pytest.fixture(scope="session")
def scene():
    return make_scene()


@pytest.fixture(scope="session")
def text_img():
    return make_text()


@pytest.fixture
def scene_file(tmp_path, scene):
    p = tmp_path / "scene.png"
    cv2.imwrite(str(p), scene)
    return str(p)
