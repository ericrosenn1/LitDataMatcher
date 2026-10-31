import json
import sys

import pytest

from litdatamatcher.development_campaign import resume_campaign


def test_campaign_resume_consumes_state_verifies_artifacts_and_continues(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    plan = {
        "campaign_id": "test",
        "milestones": [
            {
                "id": "first",
                "cwd": str(tmp_path),
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('first.json').write_text('{\"ok\":true}')",
                ],
                "outputs": [str(first)],
            },
            {
                "id": "second",
                "cwd": str(tmp_path),
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('second.json').write_text(Path('first.json').read_text())",
                ],
                "outputs": [str(second)],
                "depends_on": ["first"],
            },
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    partial = resume_campaign(path, tmp_path / "state", max_jobs=1)
    assert partial["items_succeeded"] == 1
    assert not second.exists()
    finished = resume_campaign(path, tmp_path / "state")
    assert finished["task_status"] == "PASS"
    assert len(finished["executions_this_invocation"]) == 1
    first.write_text("broken")
    repaired = resume_campaign(path, tmp_path / "state")
    assert repaired["task_status"] == "PASS"
    assert json.loads(first.read_text()) == {"ok": True}
    assert len(repaired["executions_this_invocation"]) == 1
    replay = resume_campaign(path, tmp_path / "state")
    assert replay["executions_this_invocation"] == []
    plan["milestones"][0]["command"][-1] += "\nprint('changed')"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan changed"):
        resume_campaign(path, tmp_path / "state")


def test_campaign_failed_milestone_has_bounded_resume(tmp_path):
    plan = {
        "campaign_id": "test-failure",
        "milestones": [
            {
                "id": "a",
                "cwd": str(tmp_path),
                "command": [sys.executable, "-c", "raise SystemExit(4)"],
                "outputs": [str(tmp_path / "never.json")],
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    for _ in range(3):
        state = resume_campaign(path, tmp_path / "state")
        assert state["task_status"] == "FAIL"
    final = resume_campaign(path, tmp_path / "state")
    assert final["last_execution"]["status"] == "IDLE"
