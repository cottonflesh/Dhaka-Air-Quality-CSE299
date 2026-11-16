from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
APP = ROOT / "week8" / "app.py"
TEMPLATE_DIR = ROOT / "week8" / "templates"
HTML_SOURCE = ROOT / "week9_dashboard.html"
HTML_TARGET = TEMPLATE_DIR / "week9_dashboard.html"
PATCH_SOURCE = ROOT / "week9_api_patch.py"
MARKER = "# WEEK9_DASHBOARD_API"

if not APP.exists():
    raise SystemExit("Cannot find week8/app.py. Run this installer from the repository root.")
if not HTML_SOURCE.exists() or not PATCH_SOURCE.exists():
    raise SystemExit("week9_dashboard.html or week9_api_patch.py is missing.")

app_text = APP.read_text(encoding="utf-8")
if MARKER not in app_text:
    required = ["ROOT =", "app = FastAPI", "from functools import lru_cache",
                "import numpy as np", "import pandas as pd"]
    missing = [item for item in required if item not in app_text]
    if missing:
        raise SystemExit("Week 8 app does not match expected structure: " + ", ".join(missing))
    patch = PATCH_SOURCE.read_text(encoding="utf-8")
    APP.write_text(app_text.rstrip() + "\n\n" + patch.strip() + "\n", encoding="utf-8")
    print("Added Week 9 API endpoints to week8/app.py.")
else:
    print("Week 9 API endpoints already exist; app.py left unchanged.")

TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(HTML_SOURCE, HTML_TARGET)
print("Installed week8/templates/week9_dashboard.html.")
print("Run the Week 8 app as usual, then open http://127.0.0.1:8000/week9")
