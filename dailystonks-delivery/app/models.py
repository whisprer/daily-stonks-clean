from __future__ import annotations

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base

def _utcnow() -> datetime:
    return datetime.utcnow()

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    tier: Mapped[str] = mapped_column(String(32), default="FREE", index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    preference_token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    targets: Mapped[list["DeliveryTarget"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    selections: Mapped[list["CardSelection"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    schedules: Mapped[list["DeliverySchedule"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    entitlements: Mapped[list["Entitlement"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class DeliveryTarget(Base):
    __tablename__ = "delivery_targets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    email: Mapped[str] = mapped_column(String(320), index=True)
    label: Mapped[str] = mapped_column(String(64), default="primary")
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    unsubscribed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped[User] = relationship(back_populates="targets")
    schedules: Mapped[list["DeliverySchedule"]] = relationship(back_populates="target", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("user_id", "email", name="uq_target_user_email"),)

class CardSelection(Base):
    __tablename__ = "card_selections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    card_id: Mapped[str] = mapped_column(String(64), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped[User] = relationship(back_populates="selections")

    __table_args__ = (UniqueConstraint("user_id", "card_id", name="uq_selection_user_card"),)

class DeliverySchedule(Base):
    __tablename__ = "delivery_schedules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("delivery_targets.id", ondelete="CASCADE"), index=True)

    timezone: Mapped[str] = mapped_column(String(64), default="Europe/London")
    send_time_hhmm: Mapped[str] = mapped_column(String(5), default="06:30")  # local time, 24h
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    next_run_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    override_regen: Mapped[bool] = mapped_column(Boolean, default=False)
    override_exclude: Mapped[str | None] = mapped_column(Text, nullable=True)
    override_include: Mapped[str | None] = mapped_column(Text, nullable=True)
    override_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped[User] = relationship(back_populates="schedules")
    target: Mapped[DeliveryTarget] = relationship(back_populates="schedules")
    runs: Mapped[list["DeliveryRun"]] = relationship(back_populates="schedule", cascade="all, delete-orphan")

class DeliveryRun(Base):
    __tablename__ = "delivery_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    schedule_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("delivery_schedules.id", ondelete="CASCADE"), index=True)
    run_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)  # PENDING|SENT|FAILED|SKIPPED
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    report_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    chosen_keys: Mapped[str | None] = mapped_column(Text, nullable=True)
    subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    to_email: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    schedule: Mapped[DeliverySchedule] = relationship(back_populates="runs")

    __table_args__ = (UniqueConstraint("schedule_id", "run_at", name="uq_run_schedule_runat"),)

class PaymentEvent(Base):
    __tablename__ = "payment_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider: Mapped[str] = mapped_column(String(16), default="PAYPAL", index=True)
    event_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(128), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    raw_json: Mapped[str] = mapped_column(Text)

class Entitlement(Base):
    __tablename__ = "entitlements"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True)
    tier: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE", index=True)  # ACTIVE|CANCELLED|EXPIRED
    valid_from: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="PAYPAL")

    user: Mapped[User] = relationship(back_populates="entitlements")

    __table_args__ = (UniqueConstraint("user_id", "tier", "status", "valid_from", name="uq_entitlement_user_tier_status_from"),)
