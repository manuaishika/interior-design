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
    DateTime, ForeignKey, Integer, LargeBinary, String, Text, inspect, select,
    text,
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
    # Null for a design kept outside any conversation — the common case today.
    # Set when it was kept from inside a thread, so reopening the thread shows
    # what was actually drawn while talking about it.
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True,
        default=None)
    title: Mapped[str] = mapped_column(String(160), default="")
    room: Mapped[str] = mapped_column(String(60), default="")
    style: Mapped[str] = mapped_column(String(60), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    image: Mapped[bytes] = mapped_column(LargeBinary)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    default=now, index=True)


class Conversation(Base):
    """A thread about one room: the photo it started from, and everything
    said about it since. Designs kept while the thread is open hang off it
    (see Design.conversation_id), so reopening one shows what was drawn too.
    """
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(160), default="")
    room: Mapped[str] = mapped_column(String(60), default="")
    style: Mapped[str] = mapped_column(String(60), default="")
    # The original room photo, so a thread can be reopened and talked about
    # again without asking the person to upload it a second time.
    photo: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    default=now, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                     default=now, index=True)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))          # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    default=now)


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


def _add_missing_columns(sync_conn) -> None:
    """create_all only creates tables that do not exist yet — it never alters
    one that does. `designs` already existed, with real people's designs in
    it, by the time conversation_id was added to the model. This is the
    one-line version of a migration for a project with one schema and, until
    now, no history to migrate (see HANDOFF.md); if more of these
    accumulate, that line is where this becomes Alembic.
    """
    inspector = inspect(sync_conn)
    if "designs" not in inspector.get_table_names():
        return   # create_all just made it fresh, already with the column
    existing = {c["name"] for c in inspector.get_columns("designs")}
    if "conversation_id" in existing:
        return
    # No FK clause on SQLite: it does not enforce one without a pragma
    # nothing here turns on, and ALTER TABLE ... ADD CONSTRAINT is not
    # supported there anyway. Postgres gets the real constraint.
    if sync_conn.dialect.name == "postgresql":
        sync_conn.execute(text(
            "ALTER TABLE designs ADD COLUMN conversation_id INTEGER "
            "REFERENCES conversations(id) ON DELETE CASCADE"))
    else:
        sync_conn.execute(text(
            "ALTER TABLE designs ADD COLUMN conversation_id INTEGER"))


async def create_tables() -> None:
    if _engine is None:
        raise RuntimeError("store.configure() was never called")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


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
                      note: str = "", width: int = 0, height: int = 0,
                      conversation_id: int | None = None) -> Design:
    # conversation_id is trusted here — the caller must already have checked
    # it belongs to user_id (conversation_for does that), the same way every
    # other write in this module leaves ownership to its caller's lookup.
    design = Design(user_id=user_id, image=image, title=title[:160],
                    room=room[:60], style=style[:60], note=note,
                    width=width, height=height, conversation_id=conversation_id)
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


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def _title_for(first_message: str, room: str, style: str) -> str:
    """Never blank — a list of "Untitled" is not a history.

    The person's own words, if there are any yet, because that is what they
    will recognise the thread by later. Otherwise what the thread is *of*,
    which is still more useful than a number.
    """
    first_message = (first_message or "").strip()
    if first_message:
        return first_message[:60]
    room = (room or "a room").strip().replace("-", " ").title() or "A room"
    style = (style or "").strip().replace("-", " ").title()
    stamp = now().strftime("%d %b")
    return f"{room} · {style} · {stamp}" if style else f"{room} · {stamp}"


async def start_conversation(db: AsyncSession, user_id: int, photo: bytes, *,
                             room: str = "", style: str = "",
                             first_message: str = "") -> Conversation:
    conv = Conversation(
        user_id=user_id, photo=photo, room=room[:60], style=style[:60],
        title=_title_for(first_message, room, style),
    )
    db.add(conv)
    await db.commit()
    return conv


async def conversations_for(db: AsyncSession, user_id: int,
                            limit: int = 60) -> list[Conversation]:
    rows = await db.scalars(
        select(Conversation).where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc()).limit(limit)
    )
    return list(rows)


async def conversation_for(db: AsyncSession, user_id: int,
                           conversation_id: int) -> Conversation | None:
    """Always scoped to the owner, so an id from someone else's thread
    returns nothing rather than their photo and messages."""
    return await db.scalar(
        select(Conversation).where(Conversation.id == conversation_id,
                                   Conversation.user_id == user_id))


async def messages_for(db: AsyncSession, conversation_id: int) -> list[Message]:
    rows = await db.scalars(
        select(Message).where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return list(rows)


async def designs_for_conversation(db: AsyncSession,
                                   conversation_id: int) -> list[Design]:
    rows = await db.scalars(
        select(Design).where(Design.conversation_id == conversation_id)
        .order_by(Design.created_at.desc())
    )
    return list(rows)


async def add_message(db: AsyncSession, conversation: Conversation, role: str,
                      content: str) -> Message:
    """Appends one turn and bumps updated_at, so the thread list stays sorted
    by when it was last actually talked in rather than when it was made."""
    message = Message(conversation_id=conversation.id, role=role,
                      content=content)
    db.add(message)
    conversation.updated_at = now()
    await db.commit()
    return message


async def rename_conversation(db: AsyncSession, user_id: int,
                              conversation_id: int, title: str) -> Conversation | None:
    conv = await conversation_for(db, user_id, conversation_id)
    if conv is None:
        return None
    title = title.strip()[:160]
    if title:
        conv.title = title
    await db.commit()
    return conv


async def delete_conversation(db: AsyncSession, user_id: int,
                              conversation_id: int) -> bool:
    """Deletes the thread and everything hung off it.

    `ondelete="CASCADE"` is real on Postgres but SQLite only honours it with
    `PRAGMA foreign_keys=ON`, which nothing here turns on — and this project
    runs on both. Rather than make the app's correctness depend on a pragma
    the async SQLite driver may or may not have applied, the messages and any
    designs kept from this thread are deleted explicitly, so this behaves the
    same on a laptop's SQLite file as it does in production.
    """
    conv = await conversation_for(db, user_id, conversation_id)
    if conv is None:
        return False
    for message in await messages_for(db, conversation_id):
        await db.delete(message)
    for design in await designs_for_conversation(db, conversation_id):
        await db.delete(design)
    await db.delete(conv)
    await db.commit()
    return True
