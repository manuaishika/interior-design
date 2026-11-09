"""The villa project: every photo belongs to one space, the page can find
its files on both copies, and the drawing script resumes rather than redraws."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VILLA = ROOT / "static" / "projects" / "villa"


def project():
    return json.loads((VILLA / "project.json").read_text())


def test_every_photo_is_in_exactly_one_space():
    shots = sorted(n for s in project()["spaces"] for n in s["shots"])
    assert shots == list(range(1, 47))
    for n in shots:
        assert (VILLA / "before" / f"{n:02d}.jpg").exists()


def test_every_after_listed_exists_on_both_copies():
    for n in project().get("after", []):
        assert (VILLA / "after" / f"{n:02d}.jpg").exists()
        assert (ROOT / "docs" / "static" / "projects" / "villa" / "after" / f"{n:02d}.jpg").exists()


def test_pages_copy_matches():
    docs = ROOT / "docs" / "static" / "projects" / "villa"
    assert json.loads((docs / "project.json").read_text()) == project()
    assert len(list((docs / "before").glob("*.jpg"))) == 46


def test_outside_and_fitted_rooms_are_not_fully_redesigned():
    """A full redesign plans new walls and ceilings, which outside would mean
    repainting the house, and in a bathroom or kitchen ripping out what was
    just built."""
    for s in project()["spaces"]:
        if s["room"] in ("", "bathroom", "kitchen"):
            assert s["profile"] == "restyle", s["title"]


def test_page_shows_the_project():
    for page in ("static/showcase.html", "docs/index.html"):
        text = (ROOT / page).read_text()
        assert 'id="projGrid"' in text and "static/projects/villa/" in text
        assert "Redesign in progress" in text          # until an after exists


def test_script_dry_run_needs_no_key():
    out = subprocess.run([sys.executable, "tools/draw_project.py", str(VILLA), "--dry-run"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    done = len(project().get("after", []))
    assert f"{46 - done} photo(s) to draw" in out
