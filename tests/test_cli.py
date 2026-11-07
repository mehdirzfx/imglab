import os
import subprocess
import sys

import cv2

from conftest import ROOT

SCRIPT = os.path.join(ROOT, "imglab.py")


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, timeout=180)


def test_version_and_list():
    r = run("--version")
    assert r.returncode == 0 and "imglab" in (r.stdout + r.stderr)
    r = run("--list")
    assert r.returncode == 0 and "[Edges] canny" in r.stdout and "[Plate] plate_enhance" in r.stdout


def test_status_lists_components():
    r = run("--status")
    assert r.returncode == 0
    for key in ("easyocr", "rapidocr", "contrib", "model_edsr_x4"):
        assert key in r.stdout


def test_unknown_component_is_rejected():
    assert run("--install", "nonsense").returncode == 2


def test_single_image(tmp_path, scene_file):
    out = tmp_path / "out.png"
    r = run(scene_file, "-o", str(out), "-p", "perspective:quad=auto,plate=1,aspect=4;unsharp:amount=1.5")
    assert r.returncode == 0, r.stderr
    h, w = cv2.imread(str(out)).shape[:2]
    assert abs(w / h - 4.0) < 0.1


def test_bad_pipeline_fails_cleanly(tmp_path, scene_file):
    r = run(scene_file, "-o", str(tmp_path / "o.png"), "-p", "does_not_exist")
    assert r.returncode != 0


def test_batch_with_preset(tmp_path, scene):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"img{i}.png"), scene)
    (src / "notes.txt").write_text("ignored")
    sys.path.insert(0, ROOT)
    import imglab as L
    preset = tmp_path / "p.json"
    L.save_preset(str(preset), [("perspective", {"quad": "auto", "aspect": 4.0}), ("unsharp", {"amount": 1.0})])
    r = run(str(src), "-o", str(dst), "--preset", str(preset), "--auto-persp")
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(dst)) == ["img0_out.png", "img1_out.png", "img2_out.png"]


def test_imports_without_tkinter():
    """The CLI must keep working on machines (and CI runners) that have no tkinter."""
    code = f"import sys; sys.modules['tkinter'] = None; sys.path.insert(0, {ROOT!r}); import imglab; print(imglab.HAVE_TK)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_analyze_prints_a_pipeline(tmp_path, scene):
    p = tmp_path / "t.png"
    cv2.imwrite(str(p), cv2.resize(cv2.resize(scene, (80, 60)), (640, 480), interpolation=cv2.INTER_NEAREST))
    r = subprocess.run([sys.executable, SCRIPT, str(p), "--analyze"], capture_output=True, text=True)
    assert r.returncode == 0 and "Suggested pipeline" in r.stdout and "depixelate" in r.stdout


def test_compare_flag_writes_extra_image(tmp_path, scene_file):
    out = tmp_path / "o.png"
    r = subprocess.run([sys.executable, SCRIPT, scene_file, "-o", str(out), "-p", "unsharp", "--compare"], capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "o_compare.png").is_file()
