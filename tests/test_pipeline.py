import json

import numpy as np
import pytest

import imglab as L


def test_parse_pipeline():
    steps = L.parse_pipeline("perspective:quad=auto,plate=1,aspect=4; unsharp:amount=1.5 ;")
    assert steps == [("perspective", {"quad": "auto", "plate": "1", "aspect": "4"}), ("unsharp", {"amount": "1.5"})]


def test_unknown_operation_and_parameter():
    with pytest.raises(KeyError):
        L.parse_pipeline("does_not_exist")
    with pytest.raises(KeyError):
        L.coerce(L.OPS["unsharp"], {"nope": "1"})


def test_coerce_types():
    vals = L.coerce(L.OPS["canny"], {"auto": "0", "low": "20", "sigma": "0.5"})
    assert vals["auto"] is False and vals["low"] == 20 and vals["sigma"] == 0.5
    assert vals["blur"] == 5  # untouched default


def test_run_pipeline(scene):
    out = L.run_pipeline(scene, L.parse_pipeline("perspective:quad=auto,plate=1,aspect=4;unsharp:amount=1"))
    assert out.shape[1] / out.shape[0] == pytest.approx(4.0, abs=0.1)


def test_recipe_string_round_trips(scene):
    recipe = [("canny", {"auto": True, "low": 40, "sigma": 0.25}), ("gamma", {"gamma": 1.5})]
    steps = L.parse_pipeline(L.recipe_string(recipe))
    assert np.array_equal(L.run_pipeline(scene, steps), L.run_pipeline(scene, [(n, {k: str(v) for k, v in p.items()}) for n, p in recipe]))


def test_preset_round_trip(tmp_path):
    steps = [("perspective", {"quad": "auto", "aspect": 4.0}), ("unsharp", {"amount": 1.2})]
    p = tmp_path / "preset.json"
    L.save_preset(str(p), steps)
    assert json.loads(p.read_text())["version"] == 1
    assert L.load_preset(str(p)) == steps


def test_preset_with_unknown_operation_is_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"version": 1, "steps": [["nope", {}]]}))
    with pytest.raises(KeyError):
        L.load_preset(str(p))


def test_batch_steps_redetects_perspective():
    steps = L.batch_steps([("perspective", {"quad": "1|2|3|4|5|6|7|8", "aspect": 4.0})], True)
    assert steps[0][1]["quad"] == "auto" and steps[0][1]["plate"] is True
    kept = L.batch_steps([("perspective", {"quad": "1|2|3|4|5|6|7|8"})], False)
    assert kept[0][1]["quad"] == "1|2|3|4|5|6|7|8"


def test_unicode_paths(tmp_path, scene):
    p = tmp_path / "عکس پلاک.png"
    L.imwrite(str(p), scene)
    assert L.imread(str(p)).shape == scene.shape
