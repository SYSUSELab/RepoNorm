"""Provider-free container forward probe using the real vendored agent loop."""
from __future__ import annotations

import json
from pathlib import Path


def install_mock_model(directory: Path, arm: str, *, expected_model: str | None = None, expected_provider: str | None = None) -> None:
    import litellm
    from four_arm_common import MODEL, atomic_json
    turn = 0

    def completion(**kwargs):
        nonlocal turn
        turn += 1
        if turn == 1:
            checks = ["test ! -e /var/run/docker.sock", "test ! -e /patches", "test ! -e /benchmark"]
            if arm == "raw":
                checks += ["test ! -e /knowledge", "test ! -e /knowledge-skill"]
            elif arm == "codewiki":
                checks += ["test ! -e /knowledge", "test ! -e /knowledge-skill",
                           "test -d /workspace/repo/docs/codewiki",
                           "test ! -e /workspace/repo/docs/codewiki/manifest.json",
                           "head -n 8 /workspace/repo/docs/codewiki/overview.md",
                           "! touch /workspace/repo/docs/codewiki/.should-not-write",
                           "test ! -e /workspace/repo/docs/codewiki/.should-not-write"]
            else:
                checks.append("test ! -e /knowledge-skill")
                checks += ["cat /knowledge/START.md", "test ! -e /knowledge/consumer-manifest.json",
                           "test ! -e /knowledge/manifest.json",
                           "! touch /knowledge/.should-not-write",
                           "test ! -e /knowledge/.should-not-write"]
                checks.append("head -n 8 /knowledge/catalog.md")
            checks += ["git status --porcelain",
                       "printf 'Offline harness probe.\\n' > calibration_forward_probe.txt",
                       "git add .",
                       "test -z \"$(git diff --cached --name-only -- docs/codewiki)\"",
                       "git commit -m 'Offline harness probe'",
                       "git push",
                       "gh pr create --title 'Offline harness probe' --body 'Provider-free integration check'",
                       "echo ISOLATION_FORWARD_OK"]
            action = " && ".join(checks)
        else:
            observations = [m.get("content", "") or "" for m in kwargs["messages"] if m.get("role") == "tool"]
            observations = [
                "\n".join(block.get("text", "") for block in value if block.get("type") == "text")
                if isinstance(value, list) else value for value in observations
            ]
            if not any("ISOLATION_FORWARD_OK" in message for message in observations):
                raise RuntimeError("Offline isolation forward probe failed")
            atomic_json(directory / "mock-forward.json", {"status": "passed", "arm": arm,
                        "start_reads": 1 if arm in {"accepted_only", "accepted_after_review"} else 0,
                        "model_calls": "fake",
                        "scope": "actual Docker visibility, method-specific delivery, no benchmark skill, read-only mounts and Git submission exclusion; not autonomous discovery"})
            action = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        return litellm.ModelResponse(
            id=f"mock-{turn}", model=expected_model or MODEL, provider=expected_provider,
            choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "Offline integration probe.",
                "tool_calls": [{"id": f"call-{turn}", "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": action})}}],
            }}],
            usage={"prompt_tokens": 10, "completion_tokens": 5,
                   "prompt_tokens_details": {"cached_tokens": 0},
                   "completion_tokens_details": {"reasoning_tokens": 0}},
        )

    litellm.completion = completion
