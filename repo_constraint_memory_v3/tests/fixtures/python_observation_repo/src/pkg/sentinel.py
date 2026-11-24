from pathlib import Path

Path("OBSERVATION_EXECUTED").write_text("unsafe", encoding="utf-8")
raise RuntimeError("this fixture must never be imported or executed")
