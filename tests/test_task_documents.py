"""Offline source/mount/submission checks; no model or Docker dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from four_arm_worker import task_prompt
from task_documents import (
    MOUNTED_CHECK_SCRIPT, ROOT, UNMOUNTED_CHECK_SCRIPT, assert_no_document_patch,
    protect_mounted_documents, verify_unmounted_image,
)


class TaskDocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        (self.root / "docs").mkdir()
        (self.root / "docs/original.md").write_text("Original source documentation\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root)

    def script(self, script, *args):
        return subprocess.run([sys.executable, "-c", script, *args], cwd=self.root,
                              capture_output=True, text=True)

    def documents(self):
        directory = self.root / "docs/codewiki"
        directory.mkdir()
        (directory / "overview.md").write_text("Original generated documentation\n")
        return directory, {"overview.md": hashlib.sha256((directory / "overview.md").read_bytes()).hexdigest()}

    def test_codewiki_has_only_location_hint_and_n4c_has_no_skill(self):
        (self.root / "feature.md").write_text("Exact task\n")
        self.assertEqual(task_prompt(self.root, "codewiki"), task_prompt(self.root, "raw") +
                         "\n\nAdditional repository documentation is available under `docs/codewiki/`; consult relevant pages when useful.\n")
        self.assertNotIn("knowledge", task_prompt(self.root, "codewiki"))
        for arm in ("accepted_only", "accepted_after_review"):
            prompt = task_prompt(self.root, arm)
            self.assertTrue(prompt.startswith("Exact task\n\n\n"))
            for instruction in ("Before editing", "/knowledge/START.md once",
                                "rules for the affected", "relative Markdown links",
                                "directory of the page containing them",
                                "relevant part files", "repository-wide and cross-file",
                                "scope, conditions, and exceptions", "reference_only"):
                self.assertIn(instruction, prompt)
            self.assertNotIn("skill", prompt.lower())
            self.assertNotIn("docs/codewiki", prompt)
        self.assertEqual(task_prompt(self.root, "accepted_only"),
                         task_prompt(self.root, "accepted_after_review"))
        with self.assertRaises(ValueError):
            task_prompt(self.root, "wrong")

    def test_unmounted_probe_rejects_existing_ignored_directory_and_symlink(self):
        self.assertEqual(self.script(UNMOUNTED_CHECK_SCRIPT).returncode, 0)
        target = self.root / "docs/codewiki"
        target.mkdir()
        self.assertNotEqual(self.script(UNMOUNTED_CHECK_SCRIPT).returncode, 0)
        target.rmdir()
        target.symlink_to(self.root / "missing", target_is_directory=True)
        self.assertNotEqual(self.script(UNMOUNTED_CHECK_SCRIPT).returncode, 0)
        target.unlink()
        (self.root / "docs/original.md").unlink()
        (self.root / "docs").rmdir()
        (self.root / "docs").symlink_to(self.root, target_is_directory=True)
        self.assertNotEqual(self.script(UNMOUNTED_CHECK_SCRIPT).returncode, 0)

    def test_local_git_exclusion_preserves_original_docs_and_normal_submission(self):
        directory, hashes = self.documents()
        result = self.script(MOUNTED_CHECK_SCRIPT, json.dumps(hashes))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["document_sha256"], hashes)
        (self.root / "docs/original.md").write_text("Legitimate edited documentation\n")
        self.git("add", ".")
        changed = self.git("diff", "--cached", "--name-only").decode()
        self.assertEqual(changed, "docs/original.md\n")
        self.assertTrue((directory / "overview.md").is_file())
        if shutil.which("rg"):
            # An explicit documented limitation, not a discovery-success test.
            listed = subprocess.check_output(["rg", "--files"], cwd=self.root).decode()
            self.assertNotIn("docs/codewiki/overview.md", listed)
        found = subprocess.check_output(["find", "docs", "-name", "overview.md"], cwd=self.root).decode()
        self.assertIn("docs/codewiki/overview.md", found)
        self.assertFalse((self.root / ".gitignore").exists())
        assert_no_document_patch(self.git("diff", "--cached").decode())

    def test_forced_document_stage_is_not_silently_filtered(self):
        _, hashes = self.documents()
        self.assertEqual(self.script(MOUNTED_CHECK_SCRIPT, json.dumps(hashes)).returncode, 0)
        self.git("add", "-f", "docs/codewiki/overview.md")
        with self.assertRaisesRegex(ValueError, "readonly CodeWiki"):
            assert_no_document_patch(self.git("diff", "--cached").decode())

    def test_mounted_extra_tamper_symlink_and_tracked_path_fail_closed(self):
        directory, hashes = self.documents()
        document = directory / "overview.md"
        original = document.read_bytes()
        document.write_bytes(b"tampered")
        self.assertNotEqual(self.script(MOUNTED_CHECK_SCRIPT, json.dumps(hashes)).returncode, 0)
        document.write_bytes(original)
        extra = directory / "extra.md"
        extra.write_text("extra")
        self.assertNotEqual(self.script(MOUNTED_CHECK_SCRIPT, json.dumps(hashes)).returncode, 0)
        extra.unlink()
        extra.symlink_to(document)
        self.assertNotEqual(self.script(MOUNTED_CHECK_SCRIPT, json.dumps(hashes)).returncode, 0)
        extra.unlink()
        self.git("add", "docs/codewiki")
        self.assertNotEqual(self.script(MOUNTED_CHECK_SCRIPT, json.dumps(hashes)).returncode, 0)

    def test_symlink_git_exclude_is_rejected_without_writing(self):
        _, hashes = self.documents()
        exclude = self.root / ".git/info/exclude"
        exclude.unlink()
        outside = self.root / "untouched"
        outside.write_text("unchanged")
        exclude.symlink_to(outside)
        self.assertNotEqual(self.script(MOUNTED_CHECK_SCRIPT, json.dumps(hashes)).returncode, 0)
        self.assertEqual(outside.read_text(), "unchanged")

    def test_native_git_parser_rejects_rename_into_delivery_and_allows_neighbor(self):
        rename = "diff --git a/docs/original.md b/docs/codewiki/copied.md\nsimilarity index 100%\nrename from docs/original.md\nrename to docs/codewiki/copied.md\n"
        with self.assertRaisesRegex(ValueError, "readonly CodeWiki"):
            assert_no_document_patch(rename)
        rename_out = "diff --git a/docs/codewiki/copied.md b/docs/original.md\nsimilarity index 100%\nrename from docs/codewiki/copied.md\nrename to docs/original.md\n"
        with self.assertRaisesRegex(ValueError, "readonly CodeWiki"):
            assert_no_document_patch(rename_out)
        copy_from = rename_out.replace("rename from", "copy from").replace("rename to", "copy to")
        with self.assertRaisesRegex(ValueError, "readonly CodeWiki"):
            assert_no_document_patch(copy_from)
        assert_no_document_patch(rename.replace("docs/codewiki/", "docs/codewiki-other/"))

    def test_patch_parser_does_not_inherit_a_subdirectory_prefix(self):
        rename = "diff --git a/docs/original.md b/docs/other.md\nsimilarity index 100%\nrename from docs/original.md\nrename to docs/other.md\n"
        with patch("task_documents.subprocess.run", wraps=subprocess.run) as run:
            assert_no_document_patch(rename)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.kwargs["cwd"] == ROOT for call in run.call_args_list))

    def test_original_snapshot_and_actual_mount_are_verified(self):
        expected = {"head": "a" * 40, "files": []}
        with patch("task_documents.command", return_value=json.dumps(expected)):
            self.assertTrue(verify_unmounted_image("sha256:image", expected)["destination_absent"])
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_unmounted_image("sha256:image", expected | {"head": "b" * 40})
        directory, hashes = self.documents()
        handoff = {"knowledge_directory": str(directory)}
        mount = {"Destination": "/workspace/repo/docs/codewiki", "Source": str(directory), "RW": False, "Type": "bind"}
        with patch("task_documents.command", return_value=json.dumps({"document_sha256": hashes})):
            protect_mounted_documents("container", handoff, {"Mounts": [mount]})
        for change in ({"RW": True}, {"Source": str(self.root)}, {"Destination": "/knowledge"}, {"Type": "volume"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                protect_mounted_documents("container", handoff, {"Mounts": [mount | change]})


if __name__ == "__main__":
    unittest.main()
