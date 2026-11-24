"""The site boots without its database, and the legal pages exist."""
import asyncio

from fastapi.testclient import TestClient

from app import main, store


def test_boots_when_database_is_down(monkeypatch):
    async def down():
        raise OSError("connection refused")
    monkeypatch.setattr(store, "create_tables", down)
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").status_code == 200


def test_database_errors_become_503(monkeypatch):
    from sqlalchemy.exc import OperationalError

    async def boom(*a, **k):
        raise OperationalError("select", {}, Exception("down"))
    monkeypatch.setattr(store, "create_tables", boom)
    assert main._database_down is not None
    res = asyncio.run(main._database_down(
        type("R", (), {"url": type("U", (), {"path": "/x"})()})(),
        OperationalError("select", {}, Exception("down"))))
    assert res.status_code == 503


def test_privacy_and_terms_are_served():
    with TestClient(main.app) as client:
        for path, word in (("/privacy", "Privacy"), ("/terms", "Terms of use")):
            res = client.get(path)
            assert res.status_code == 200 and word in res.text


def test_footer_links_to_them_on_both_pages():
    for page in ("static/showcase.html", "docs/index.html"):
        text = open(page).read()
        assert 'href="/privacy"' in text and 'href="/terms"' in text


def test_designs_page_is_in_the_nav_and_has_its_pictures():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for page, base in (("static/showcase.html", "static"), ("docs/index.html", "docs")):
        text = (root / page).read_text()
        assert 'id="v-designs"' in text
        assert text.count('data-go="designs"') >= 2          # nav + footer
        assert "'designs'" in text                              # routed
        for name in ("design-1-before.jpg", "design-1-after.jpg"):
            assert f"static/designs/{name}" in text
            assert (root / base / "static" / "designs" / name).exists() or \
                   (root / "static" / "designs" / name).exists()


def test_how_it_works_has_the_draggable_walkthrough_and_home_has_the_video():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for page in ("static/showcase.html", "docs/index.html"):
        text = (root / page).read_text()
        how = text[text.index('id="v-how"'):text.index('id="v-pricing"')]
        assert 'id="reelPhoto"' in how and "sliderDrag(photo)" in text
        home = text[text.index('id="v-home"'):text.index('id="v-studio"')]
        assert "static/videos/what-you-can-do.mp4" in home
        designs = text[text.index('id="v-designs"'):text.index('id="v-how"')]
        assert 'id="reel"' not in designs and 'id="featBa"' in designs
    for name in ("what-you-can-do.mp4", "what-you-can-do.jpg"):
        assert (root / "static" / "videos" / name).exists()
        assert (root / "docs" / "static" / "videos" / name).exists()


def _request():
    from starlette.requests import Request
    return Request({"type": "http", "headers": [], "client": ("1.2.3.4", 1), "method": "POST", "path": "/api/generate"})


def test_a_dead_database_does_not_stop_a_design(monkeypatch):
    """The meter and the plan lookup are soft: if the database is gone (a
    suspended free instance, a bad URL) the design is still drawn."""
    from app import config, main, store

    def broken():
        raise OSError("connection refused")

    monkeypatch.setattr(store, "is_configured", lambda: True)
    monkeypatch.setattr(store, "session", broken)
    key = asyncio.run(main._usage_gate(_request(), config.Settings(), "free"))
    assert key.startswith("ip:")
    asyncio.run(main._credit(key, 2))                       # swallowed, not raised


def test_an_unexpected_crash_comes_back_as_json():
    import json
    from app import main
    res = asyncio.run(main._anything_else(_request(), ZeroDivisionError("x")))
    assert res.status_code == 500
    assert "went wrong" in json.loads(res.body)["detail"]


def test_the_page_can_read_a_reply_that_is_not_json():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "static" / "showcase.html").read_text()
    assert "async function readJson(r)" in text
    assert "await r.json();\n      if (!r.ok) throw new Error(d.detail || 'Could not draw')" not in text
