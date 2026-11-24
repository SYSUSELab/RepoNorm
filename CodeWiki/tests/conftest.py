"""Make vendored integration tests source-local and credential-independent."""

import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "CodeWiki"))
sys.path.insert(0, str(project_root / "repo_constraint_memory_v3/src"))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
