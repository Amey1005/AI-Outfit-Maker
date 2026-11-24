from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import settings

_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pref_vector: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)  # float32[512] taste vector
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    items: Mapped[list["Item"]] = relationship(cascade="all, delete-orphan")


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    image_path: Mapped[str] = mapped_column(String(255))
    fine_type: Mapped[str] = mapped_column(String(32))
    coarse_type: Mapped[str] = mapped_column(String(16))
    colors: Mapped[list] = mapped_column(JSON)  # [{"name", "rgb", "fraction"}] largest first
    pattern: Mapped[str] = mapped_column(String(16))
    material: Mapped[str] = mapped_column(String(16))
    wear: Mapped[str] = mapped_column(String(16), default="unisex")  # menswear | womenswear | unisex
    type_confidence: Mapped[float] = mapped_column(default=1.0)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    item_ids: Mapped[list] = mapped_column(JSON)
    occasion: Mapped[str | None] = mapped_column(String(16), nullable=True)
    liked: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class WearEvent(Base):
    __tablename__ = "wear_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    item_ids: Mapped[list] = mapped_column(JSON)
    worn_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def init_db():
    Base.metadata.create_all(engine)
    # Databases created before the `wear` column existed get it added in place (existing items become unisex).
    if "wear" not in {c["name"] for c in inspect(engine).get_columns("items")}:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE items ADD COLUMN wear VARCHAR(16) DEFAULT 'unisex' NOT NULL"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
