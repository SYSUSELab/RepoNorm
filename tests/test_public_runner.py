"""Public batch completion must reflect saved execution errors, not just process exit."""
import json
import shutil
from types import SimpleNamespace

import pytest
from runner import __main__ as controller


@pytest.mark.parametrize("model", ["deepseek", "qwen", "gemini"])
def test_new_plans_do_not_inherit_historical_test_variants(tmp_path, model):
    selected = controller.tasks(["JIN1559-F07"])
    specs = controller.plan(model, list(controller.CONDITIONS), selected, tmp_path / "plan")
    assert len(specs) == 3
    for path in specs:
        spec = json.loads(path.read_text())
        assert "tests_variant" not in spec["pool"]
        assert spec["pool"] == controller.task_pool(selected[0])


@pytest.mark.parametrize("outcome, completed", [
    ("infrastructure_error", False),
    ("knowledge_isolation_error", False),
    ("resolved", True),
    ("functional_failure", True),
    ("budget_exhausted", True),
    (None, False),
])
def test_worker_receipt_controls_batch_completion(tmp_path, monkeypatch, outcome, completed):
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    if outcome:
        (tmp_path / "result.json").write_text(json.dumps({"outcome": outcome}))
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    row = controller.run_worker(spec)
    assert row["execution_completed"] is completed


def test_nonzero_worker_exit_cannot_be_hidden_by_saved_success(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text('{"outcome":"resolved"}')
    monkeypatch.setattr(controller.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=2))
    assert not controller.run_worker(tmp_path / "spec.json")["execution_completed"]


def test_test_filter_retains_implementation_and_removes_authored_tests():
    from cooperbench.eval.sandbox import _filter_test_files
    test = 'diff --git a/tests/test_feature.py b/tests/test_feature.py\n+assert True\n'
    implementation = 'diff --git a/src/feature.py b/src/feature.py\n+value = 1\n'
    assert _filter_test_files(test + implementation) == implementation


def test_coding_condition_uses_accepted_only_even_when_reviewed_view_is_first(tmp_path, monkeypatch):
    snapshot_id = "jin-896a62135bcc"
    root = controller.ROOT
    packages = json.loads((root / "data/knowledge/manifest.json").read_text())["packages"]
    selected = sorted((p for p in packages if p["method"] == "reponorm"
                       and p["snapshot_id"] == snapshot_id), key=lambda p: p["view"])
    assert selected[0]["view"] == "accepted_after_review"
    for package in selected:
        target = tmp_path / "data/knowledge" / package["archive"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "data/knowledge" / package["archive"], target)
    (tmp_path / "data/knowledge/manifest.json").write_text(json.dumps({"packages": selected}))
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    task = {"snapshot_id": snapshot_id, "snapshot": selected[0]["snapshot_commit"]}
    receipt = controller.stage_knowledge("reponorm", task, tmp_path / "staged")
    accepted = next(p for p in selected if p["view"] == "accepted_only")
    assert receipt["host_receipt"]["original_markdown_sha256"] == accepted["files"]
    assert len(list((tmp_path / "staged/claims").glob("*.md"))) == 69
