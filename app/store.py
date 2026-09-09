"""People, and the rooms they have had drawn.

Why this exists
---------------
The studio used to have a shared access code, which answers "are you allowed
in". It does not answer "who are you", and so it could not answer "where are
my designs" either. Saving work needs identity, and identity needs somewhere
to put it.

Where it lives
--------------
Postgres when `DATABASE_URL` is set, SQLite on disk otherwise. Both work; the
difference is only whether the data survives the host. On a free Render web
service the filesystem is wiped on every deploy, so SQLite there means losing
everyone's designs at the next push — set `DATABASE_URL` before anyone real
signs up.

Images are stored as bytes in the row. That is the wrong answer at a hundred
thousand designs and exactly the right one at a thousand: no bucket, no
credentials, no signed URLs, nothing else to go wrong. `image` is the only
column that would have to move to object storage later.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import secrets

from sqlalchemy import (
    DateTime, ForeignKey, Integer, LargeBinary, String, Text, select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

log = logging.getLogger(__name__)

# scrypt is in the standard library, so this needs no bcrypt, no passlib and
# no argon2 wheel on a slow free dyno. The parameters are the interactive-login
# set from the RFC; raising N is the knob if hardware moves on.
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${key.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, key_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        key = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                             **_SCRYPT)
    except (ValueError, TypeError):
        return False
    # compare_digest so a wrong password cannot be narrowed down by timing.
    return secrets.compare_digest(key.hex(), key_hex)


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Stored lowercased. Someone who signs up as Anna@ and returns as anna@ is
    # the same person and would otherwise get a second, empty account.
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    # Null for anyone who only ever signs in with Google — there is no password
    # to store, and inventing one would be worse than not having it.
    password_hash: Mapped[str | None] = mapped_column(String(255), default=None)
    google_id: Mapped[str | None] = mapped_column(String(64), unique=True,
                                                  index=True, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    default=now)


class Design(Base):
    __tablename__ = "designs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(160), default="")
    room: Mapped[str] = mapped_column(String(60), default="")
    style: Mapped[str] = mapped_column(String(60), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    image: Mapped[bytes] = mapped_column(LargeBinary)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    default=now, index=True)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engine = None
_sessions: async_sessionmaker[AsyncSession] | None = None


def _url(raw: str) -> str:
    """Normalise whatever the host handed us into an async driver URL.

    Render and Heroku both hand out `postgres://`, which SQLAlchemy has not
    accepted for years, and neither hands out the `+asyncpg` this needs. Fixing
    it here means nobody has to know that to deploy.
    """
    if not raw:
        return "sqlite+aiosqlite:///./second-draft.db"
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://"):
        raw = "postgresql+asyncpg://" + raw[len("postgresql://"):]
    if raw.startswith("sqlite://") and "+aiosqlite" not in raw:
        raw = raw.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return raw


def configure(database_url: str = "") -> None:
    global _engine, _sessions
    _engine = create_async_engine(_url(database_url), pool_pre_ping=True)
    _sessions = async_sessionmaker(_engine, expire_on_commit=False)


async def create_tables() -> None:
    if _engine is None:
        raise RuntimeError("store.configure() was never called")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def session() -> AsyncSession:
    if _sessions is None:
        raise RuntimeError("store.configure() was never called")
    return _sessions()


async def dispose() -> None:
    global _engine, _sessions
    if _engine is not None:
        await _engine.dispose()
    _engine, _sessions = None, None


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

def tidy_email(email: str) -> str:
    return (email or "").strip().lower()


async def by_email(db: AsyncSession, email: str) -> User | None:
    return await db.scalar(select(User).where(User.email == tidy_email(email)))


async def by_id(db: AsyncSession, user_id: int) -> User | None:
    return await db.get(User, user_id)


async def sign_up(db: AsyncSession, email: str, password: str,
                  name: str = "") -> User:
    email = tidy_email(email)
    if await by_email(db, email):
        raise ValueError("There is already an account with that email.")
    user = User(email=email, name=name.strip(),
                password_hash=hash_password(password))
    db.add(user)
    await db.commit()
    return user


async def sign_in(db: AsyncSession, email: str, password: str) -> User | None:
    user = await by_email(db, email)
    # Hash regardless of whether the account exists, so the time taken does not
    # reveal which emails are registered.
    stored = user.password_hash if user and user.password_hash else hash_password("x")
    ok = verify_password(password, stored)
    return user if (ok and user and user.password_hash) else None


async def from_google(db: AsyncSession, google_id: str, email: str,
                      name: str = "") -> User:
    """Find or make the account behind a Google sign-in.

    Matching on the email as well as the Google id is what lets someone who
    signed up with a password later click "Continue with Google" and land in
    the same account instead of a second empty one.
    """
    email = tidy_email(email)
    user = await db.scalar(select(User).where(User.google_id == google_id))
    if user is None:
        user = await by_email(db, email)
    if user is None:
        user = User(email=email, name=name.strip(), google_id=google_id)
        db.add(user)
    else:
        user.google_id = google_id
        if not user.name and name:
            user.name = name.strip()
    await db.commit()
    return user


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------

async def save_design(db: AsyncSession, user_id: int, image: bytes, *,
                      title: str = "", room: str = "", style: str = "",
                      note: str = "", width: int = 0, height: int = 0) -> Design:
    design = Design(user_id=user_id, image=image, title=title[:160],
                    room=room[:60], style=style[:60], note=note,
                    width=width, height=height)
    db.add(design)
    await db.commit()
    return design


async def designs_for(db: AsyncSession, user_id: int,
                      limit: int = 60) -> list[Design]:
    rows = await db.scalars(
        select(Design).where(Design.user_id == user_id)
        .order_by(Design.created_at.desc()).limit(limit)
    )
    return list(rows)


async def design_for(db: AsyncSession, user_id: int, design_id: int) -> Design | None:
    """Always scoped to the owner, so an id from someone else's collection
    returns nothing rather than their picture."""
    return await db.scalar(
        select(Design).where(Design.id == design_id, Design.user_id == user_id))


async def delete_design(db: AsyncSession, user_id: int, design_id: int) -> bool:
    design = await design_for(db, user_id, design_id)
    if design is None:
        return False
    await db.delete(design)
    await db.commit()
    return True
