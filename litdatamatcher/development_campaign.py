"""Resume an explicitly authored local milestone plan through the native controller.

Plans are local execution configuration, never fetched source content. This module
does not invent development work or claim a scheduler is active after it exits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .controller import Controller, integration_lease
from .data_plane import atomic_json, digest
from .strict_artifacts import strict_json_loads
from .workflow_checkpoint import file_identity


def resume_campaign(plan_path, state_root, *, max_jobs=10, timeout=1800):
    plan_path, state_root = Path(plan_path).resolve(), Path(state_root).resolve()
    plan = strict_json_loads(plan_path.read_text(encoding="utf-8-sig"))
    if not isinstance(plan.get("campaign_id"), str) or not plan["campaign_id"]:
        raise ValueError("A campaign_id is required")
    milestones = plan.get("milestones", [])
    if not isinstance(milestones, list) or not 1 <= len(milestones) <= 100:
        raise ValueError("A campaign needs 1..100 executable milestones")
    if type(max_jobs) is not int or not 1 <= max_jobs <= 100 or not 1 <= timeout <= 7200:
        raise ValueError("Use bounded job/time limits")
    ids = [m["id"] for m in milestones]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Milestone IDs must be unique nonempty strings")
    seen = set()
    for milestone in milestones:
        if not set(milestone.get("depends_on", [])) <= seen:
            raise ValueError("Milestones must be in dependency order")
        seen.add(milestone["id"])
        cwd = Path(milestone["cwd"]).resolve(strict=True)
        commit = milestone.get("required_commit")
        if commit:
            check = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                cwd=cwd,
                capture_output=True,
                check=False,
            )
            if check.returncode:
                raise ValueError(f"Required milestone commit is not an ancestor: {commit}")
    identity = digest([plan, [file_identity(path) for path in plan.get("inputs", [])]])
    state_path = state_root / "task_state.json"
    with integration_lease(state_root / "campaign-owner", plan["campaign_id"]):
        if state_path.exists():
            old = strict_json_loads(state_path.read_text(encoding="utf-8"))
            if old.get("plan_digest") != identity:
                raise ValueError(
                    "Campaign plan changed; preserve this state and select a new state root"
                )
        controller = Controller(state_root / "controller")
        try:
            if controller.mode() != "RUNNING":
                raise ValueError(
                    f"Campaign queue is {controller.mode()}; explicitly resume its controller first"
                )
            executions = []
            for _ in range(max_jobs + 1):
                with integration_lease(controller.root, "campaign-verify"):
                    controller.repair()
                    with controller.db:
                        controller.db.execute(
                            "UPDATE jobs SET status='QUEUED',not_before=0 WHERE status='FAILED' AND attempts<3"
                        )
                jobs = {row["id"]: row for row in controller.jobs()}
                completed = {key for key, row in jobs.items() if row["status"] == "SUCCEEDED"}
                eligible = [
                    m
                    for m in milestones
                    if m["id"] not in completed and set(m.get("depends_on", [])) <= completed
                ]
                for milestone in eligible:
                    controller.register(
                        milestone["id"],
                        milestone["command"],
                        milestone["cwd"],
                        milestone["outputs"],
                        retryable_exit_codes=milestone.get("retryable_exit_codes", []),
                        max_retries=2,
                    )
                jobs = {row["id"]: row for row in controller.jobs()}
                state = {
                    "campaign_id": plan["campaign_id"],
                    "plan_digest": identity,
                    "plan_path": str(plan_path),
                    "task_status": "PASS" if len(completed) == len(milestones) else "IN_PROGRESS",
                    "milestones": [
                        {
                            "id": m["id"],
                            "status": jobs.get(m["id"], {}).get("status", "WAITING_DEPENDENCY"),
                            "required_commit": m.get("required_commit"),
                            "outputs": m["outputs"],
                        }
                        for m in milestones
                    ],
                    "items_requested": len(milestones),
                    "items_succeeded": len(completed),
                    "executions_this_invocation": executions,
                    "scientific_assumptions_changed": False,
                    "background_execution_claimed": False,
                    "next_actions": [m["id"] for m in eligible],
                }
                atomic_json(state_path, state)
                if len(completed) == len(milestones) or len(executions) >= max_jobs:
                    return state
                result = controller.run_next(timeout=timeout)
                executions.append(result)
                if result.get("status") in {
                    "FAILED",
                    "RECOVERY_REQUIRED",
                    "IDLE",
                    "PAUSED_BY_USER",
                }:
                    state["task_status"] = (
                        "BLOCKED" if result["status"] == "RECOVERY_REQUIRED" else "FAIL"
                    )
                    state["last_execution"] = result
                    atomic_json(state_path, state)
                    return state
            return state
        finally:
            controller.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    state = resume_campaign(
        args.plan, args.state_root, max_jobs=args.max_jobs, timeout=args.timeout
    )
    print(json.dumps(state, indent=2))
    return 0 if state["task_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
