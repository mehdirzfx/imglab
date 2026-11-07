"""GUI smoke tests. They need tkinter and a display (CI: `xvfb-run -a pytest`); otherwise they are skipped."""
import os
import sys
import time

import pytest

tk = pytest.importorskip("tkinter")
import imglab as L  # noqa: E402

if sys.platform != "linux" and not os.environ.get("IMGLAB_GUI_TESTS"):
    pytest.skip("GUI tests run on Linux (xvfb) by default; set IMGLAB_GUI_TESTS=1 to force", allow_module_level=True)


class Ev:
    def __init__(self, x, y):
        self.x, self.y = x, y


def pump(app, seconds=0.4):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.02)


def wait_for(app, cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.update()
        time.sleep(0.02)
    assert cond(), "timed out"


@pytest.fixture
def app(scene_file, monkeypatch):
    try:
        a = L.App(scene_file)
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    monkeypatch.setattr(L.messagebox, "showinfo", lambda *a, **k: None)
    monkeypatch.setattr(L.messagebox, "showerror", lambda *a, **k: pytest.fail(f"error dialog: {a}"))
    a.geometry("1355x730+0+0")
    pump(a, 0.8)
    yield a
    a.destroy()


def apply_plate_perspective(app):
    app.persp.plate.set(True)
    app.auto_quad()
    assert len(app.pts) == 4
    app.persp.aspect.set(4.0)
    app.persp.go(True)
    wait_for(app, lambda: app.current.shape[0] < 200)


def test_default_window_size(app):
    assert (app.winfo_width(), app.winfo_height()) == (1355, 730)


def test_apply_undo_redo(app):
    before = app.current.copy()
    app.panels["Edges"].go(True)
    wait_for(app, lambda: app.recipe)
    assert [n for n, _ in app.recipe] == ["canny"]
    app.undo()
    assert (app.current == before).all() and not app.recipe
    app.redo()
    assert [n for n, _ in app.recipe] == ["canny"]


def test_split_view_with_same_size_uses_one_view(app):
    app.split_on.set(True)
    app.on_split()
    assert not app._diff()
    z = app.zoom
    app._zoom_at(100, 300, 2.0)  # even over the left half the shared view changes
    assert app.zoom != z


def test_split_view_panes_zoom_and_pan_independently(app):
    apply_plate_perspective(app)
    app.split_on.set(True)
    app.on_split()
    assert app._diff()
    sx = int(app.split_pos * app.canvas.winfo_width())
    z_e, off_e, z_o = app.zoom, list(app.off), app.zoom_o

    app._zoom_at(sx // 2, 300, 2.0)  # left pane = original
    assert app.zoom_o != z_o and app.zoom == z_e and app.off == off_e
    z_o2 = app.zoom_o
    app._zoom_at(sx + 100, 300, 1.5)  # right pane = edited
    assert app.zoom != z_e and app.zoom_o == z_o2

    off_e2, off_o2 = list(app.off), list(app.off_o)
    app._press(Ev(sx // 2, 300))
    app._drag(Ev(sx // 2 + 40, 320))
    app._release(None)
    assert app.off_o != off_o2 and app.off == off_e2

    app.fit()  # Fit resets both panes
    assert (app.zoom_o, app.off_o) != (z_o2, off_o2)


def test_hold_to_compare_and_split_handle(app):
    apply_plate_perspective(app)
    app.split_on.set(True)
    app.on_split()
    cw = app.canvas.winfo_width()
    sx = int(app.split_pos * cw)
    app._press(Ev(sx, 300))
    app._drag(Ev(int(cw * 0.3), 300))
    app._release(None)
    assert app.split_pos == pytest.approx(0.3, abs=0.01)
    app.set_compare(True)
    app.set_compare(False)


def test_pipeline_rebuild_and_reorder(app):
    apply_plate_perspective(app)
    app.panels["Enhance"].op_var.set("Gamma")
    app.panels["Enhance"].rebuild()
    app.panels["Enhance"].go(True)
    wait_for(app, lambda: len(app.recipe) == 2)
    app.pipeline.lb.selection_set(1)
    app.pipeline.up()
    assert [n for n, _ in app.recipe] == ["gamma", "perspective"]


def test_setup_dialog_lists_components(app):
    app.open_setup()
    pump(app, 0.2)
    assert set(app._setup.rows) == set(L.COMPONENTS)
    app.open_setup()  # a second click reuses the window
    assert len([w for w in app.winfo_children() if isinstance(w, L.SetupDialog)]) == 1


def test_frozen_build_marks_packages_as_source_only(app, monkeypatch):
    monkeypatch.setattr(L, "FROZEN", True)
    app.open_setup()
    d = app._setup
    d.refresh()
    for key, c in L.COMPONENTS.items():
        status = c["check"]()
        if "pip" in c and status != "ok":
            assert d.rows[key][0].cget("text") == "Source install only"
            assert str(d.rows[key][1].cget("state")) == "disabled"


def test_ocr_without_engine_offers_setup(app, monkeypatch):
    monkeypatch.setattr(L, "ocr_engines", lambda: [])
    asked = []
    monkeypatch.setattr(L.messagebox, "askyesno", lambda *a, **k: asked.append(a) or True)
    app.do_ocr()
    assert asked and app._setup.winfo_exists()


def test_auto_fix_button_repairs_a_mosaic(app):
    import cv2
    h, w = app.current.shape[:2]
    app.current = cv2.resize(cv2.resize(app.current, (w // 8, h // 8), interpolation=cv2.INTER_AREA),
                             (w // 8 * 8, h // 8 * 8), interpolation=cv2.INTER_NEAREST)
    before = app.current.copy()
    app.do_autofix()
    wait_for(app, lambda: app.recipe)
    pump(app, 0.3)
    assert app.recipe[-1][0] == "depixelate" and "pixelated" in app.msg.get()
    assert (app.current != before).any()
