"""Sign-up asks whether this is a customer or a firm.

Every account gives a name; a firm also gives a company name. No phone number.
"""

import pytest

from tests.test_accounts import client, db, join, settings  # noqa: F401  (fixtures)


def signup(client, **kw):
    data = {"email": "f@firm.test", "password": "a-good-password", "name": "Fatima"}
    data.update(kw)
    return client.post("/api/signup", data=data)


class TestWhoIsSigningUp:
    def test_a_customer_is_the_default(self, client):
        card = signup(client).json()["account"]
        assert card["kind"] == "customer" and card["company"] == ""

    def test_a_firm_gives_its_company_name(self, client):
        card = signup(client, kind="firm", company="Eleganza Interiors").json()["account"]
        assert card["kind"] == "firm" and card["company"] == "Eleganza Interiors"
        assert client.get("/api/session").json()["account"]["company"] == "Eleganza Interiors"

    def test_a_firm_without_a_company_name_is_refused(self, client):
        assert signup(client, kind="firm", company=" ").status_code == 400

    def test_a_name_is_required_for_everyone(self, client):
        assert signup(client, name="  ").status_code == 400

    def test_an_unknown_kind_is_refused(self, client):
        assert signup(client, kind="agency").status_code == 400

    def test_a_customer_cannot_smuggle_in_a_company(self, client):
        card = signup(client, kind="customer", company="Sneaky Ltd").json()["account"]
        assert card["company"] == ""


class TestGoogleCarriesTheAnswer:
    def test_the_choice_rides_a_short_lived_cookie_to_the_callback(self, client, monkeypatch):
        import app.main as main
        monkeypatch.setattr(main.google_login, "configured", lambda s: True)
        monkeypatch.setattr(main.google_login, "start_url",
                            lambda s, uri, state: "https://accounts.example/auth")
        r = client.get("/api/auth/google?kind=firm&company=Eleganza", follow_redirects=False)
        assert r.status_code in (302, 307)
        assert "sd_google_who" in r.headers.get("set-cookie", "")

    def test_a_new_google_account_gets_the_kind_an_old_one_keeps_its_own(self):
        import asyncio
        from app import store

        async def go(tmp):
            store.configure(f"sqlite+aiosqlite:///{tmp}/k.db")
            await store.create_tables()
            async with store.session() as db:
                a = await store.from_google(db, "g1", "a@x.test", "A", "firm", "Eleganza")
                b = await store.from_google(db, "g1", "a@x.test", "A", "customer", "")
                assert (a.kind, a.company) == ("firm", "Eleganza")
                assert (b.kind, b.company) == ("firm", "Eleganza")
            await store.dispose()

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(go(tmp))


class TestAnOldDatabaseGainsTheColumns:
    def test_users_made_before_kinds_existed_become_customers(self, tmp_path):
        import asyncio
        import sqlite3
        from app import store

        path = tmp_path / "old.db"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(320), "
                    "name VARCHAR(120), password_hash VARCHAR(255), google_id VARCHAR(64), "
                    "tier VARCHAR(16), created_at DATETIME)")
        con.execute("INSERT INTO users (email, name, tier) VALUES ('old@x.test','Old','free')")
        con.commit(); con.close()

        async def go():
            store.configure(f"sqlite+aiosqlite:///{path}")
            await store.create_tables()
            async with store.session() as db:
                user = await store.by_email(db, "old@x.test")
                assert user.kind == "customer" and user.company == ""
            await store.dispose()

        asyncio.run(go())
