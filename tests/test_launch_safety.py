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
