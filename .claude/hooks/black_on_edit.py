"""PostToolUse hook: run black on a .py file inside the repo after Claude writes or edits it.

Reads the hook payload from stdin. Anything that is not a repo .py file is ignored, and a
black failure (e.g. a syntax error mid-edit) never blocks the edit: the hook always exits 0.

AI assistance: drafted with Claude Code (Anthropic) from the author's specification; reviewed and tested by the author.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path:
        return
    path = Path(file_path).resolve()
    if path.suffix != ".py" or not path.is_relative_to(ROOT) or not path.exists():
        return
    subprocess.run([sys.executable, "-m", "black", "--quiet", str(path)], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
