"""Frozen CooperBench submission test-file filter."""
import re

_TEST_DIR_RE = re.compile(r"/(tests?|__tests__|spec|testdata)/")
_TEST_FILE_RE = re.compile(
    r"(^|/)("
    r"test_[^/]+"  # test_foo.py
    r"|[^/]+_test\.[A-Za-z0-9]+"  # foo_test.go / foo_test.rs / foo_test.py
    r"|[^/]+\.(test|spec)\.[A-Za-z0-9]+"  # foo.test.ts / foo.spec.js
    r"|[^/]*Test[s]?\.(java|kt|cs|scala)"  # FooTest.java
    r"|tests?\.py"  # tests.py
    r")$"
)



def _is_test_path(diff_header: str) -> bool:
    """Whether a `diff --git a/X b/Y` header names a test file, in any language."""
    paths = re.findall(r"[ab]/(\S+)", diff_header)
    return any(_TEST_DIR_RE.search("/" + p) or _TEST_FILE_RE.search(p) for p in paths)

def _filter_test_files(patch_content: str) -> str:
    """Filter test files from patch content."""
    if not patch_content:
        return patch_content

    filtered_lines = []
    skip_until_next_diff = False

    for line in patch_content.split("\n"):
        # Check if this is a new file diff header
        if line.startswith("diff --git"):
            skip_until_next_diff = _is_test_path(line)

        if not skip_until_next_diff:
            filtered_lines.append(line)

    result = "\n".join(filtered_lines)
    # Ensure patch ends with newline (required by git)
    if result and not result.endswith("\n"):
        result += "\n"
    return result
