"""Phase 4 — per-field approval state machine + cross-session persistence.

The ProCalc proposal §7 #1 says: *"the approval gate is the product, not
a polish step"*. This module is that product.

Five-state machine per AI-extracted field:

  pending         no AI suggestion yet (theoretical — we ship in ai_suggested)
  ai_suggested    extractor produced a value; awaiting builder review
  user_confirmed  builder ticked "Sight" — value accepted as-is
  user_corrected  builder edited the value; original_value preserved
  rejected        builder removed the field (e.g. spurious detection)

Calculate/Save reads from this store. It is unlocked only when every
required field is either user_confirmed or user_corrected — rejected
fields are explicit decisions and count toward unlock.

Persistence: a small JSON file per extraction_id, written to
`poc/approval_store/<extraction_id>.json`. The wizard reads it on load
and the user's per-field clicks write it on the fly. Cross-session
because the file outlives the Streamlit process.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Optional


# ---------- State enum ----------

class FieldState(str, Enum):
    PENDING = "pending"
    AI_SUGGESTED = "ai_suggested"
    USER_CONFIRMED = "user_confirmed"
    USER_CORRECTED = "user_corrected"
    REJECTED = "rejected"

    @property
    def is_resolved(self) -> bool:
        """A state counts toward the Calculate/Save unlock iff it's
        explicit — confirmed, corrected, or rejected. Pending and
        ai_suggested do NOT count."""
        return self in (
            FieldState.USER_CONFIRMED,
            FieldState.USER_CORRECTED,
            FieldState.REJECTED,
        )

    @property
    def display_badge(self) -> str:
        return {
            FieldState.PENDING:        "⏳ pending",
            FieldState.AI_SUGGESTED:   "🤖 AI suggested",
            FieldState.USER_CONFIRMED: "✅ confirmed",
            FieldState.USER_CORRECTED: "✏️ corrected",
            FieldState.REJECTED:       "❌ rejected",
        }[self]


# ---------- Per-field record ----------

@dataclass
class FieldApproval:
    """One field's approval state. `original_value` is what the AI
    suggested; `current_value` is what the user has after any edits."""
    field_key: str
    state: FieldState
    original_value: Any
    current_value: Any
    original_confidence: float
    user_note: Optional[str] = None
    updated_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "field_key": self.field_key,
            "state": self.state.value,
            "original_value": self.original_value,
            "current_value": self.current_value,
            "original_confidence": self.original_confidence,
            "user_note": self.user_note,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldApproval":
        return cls(
            field_key=d["field_key"],
            state=FieldState(d.get("state", "ai_suggested")),
            original_value=d.get("original_value"),
            current_value=d.get("current_value"),
            original_confidence=float(d.get("original_confidence", 0.0)),
            user_note=d.get("user_note"),
            updated_at=d.get("updated_at", dt.datetime.now(dt.timezone.utc).isoformat()),
        )


# ---------- Persistent store ----------

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "approval_store"


class ApprovalStore:
    """File-backed store of field approvals, keyed by extraction_id.

    Each extraction_id has its own JSON file:
      approval_store/<extraction_id>.json

    Threading note: in a Streamlit run, every user click triggers a
    full rerun on the same thread, so the in-process lock is enough.
    """

    def __init__(self, store_dir: Path = DEFAULT_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    # ---------- path / IO ----------

    def _path(self, extraction_id: str) -> Path:
        # Sanitise: extraction_ids look like "EXT-xxxxxxxx", safe filename
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in extraction_id)
        return self.store_dir / f"{safe}.json"

    def _read(self, extraction_id: str) -> dict[str, FieldApproval]:
        p = self._path(extraction_id)
        if not p.exists():
            return {}
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return {k: FieldApproval.from_dict(v) for k, v in (raw.get("fields") or {}).items()}

    def _write(self, extraction_id: str, fields: dict[str, FieldApproval]) -> None:
        p = self._path(extraction_id)
        payload = {
            "extraction_id": extraction_id,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fields": {k: v.to_dict() for k, v in fields.items()},
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---------- public API ----------

    def initialise_from_extractions(
        self,
        extraction_id: str,
        extractions: list,                # list[ExtractionField]
        *,
        force: bool = False,
    ) -> dict[str, FieldApproval]:
        """Seed the store for a new extraction. If the file already
        exists and `force=False`, this is a no-op (we preserve user state
        across reruns)."""
        with self._lock:
            existing = self._read(extraction_id)
            if existing and not force:
                # Top up any new field keys (e.g. Phase 2 fields graduated
                # from placeholder to actual) without losing user state on
                # already-tracked fields.
                changed = False
                for f in extractions:
                    if f.field_key not in existing:
                        existing[f.field_key] = FieldApproval(
                            field_key=f.field_key,
                            state=FieldState.AI_SUGGESTED,
                            original_value=f.value,
                            current_value=f.value,
                            original_confidence=f.confidence,
                        )
                        changed = True
                if changed:
                    self._write(extraction_id, existing)
                return existing

            seeded: dict[str, FieldApproval] = {}
            for f in extractions:
                seeded[f.field_key] = FieldApproval(
                    field_key=f.field_key,
                    state=FieldState.AI_SUGGESTED,
                    original_value=f.value,
                    current_value=f.value,
                    original_confidence=f.confidence,
                )
            self._write(extraction_id, seeded)
            return seeded

    def get(self, extraction_id: str) -> dict[str, FieldApproval]:
        with self._lock:
            return self._read(extraction_id)

    def update(
        self,
        extraction_id: str,
        field_key: str,
        *,
        state: Optional[FieldState] = None,
        current_value: Any = None,
        user_note: Optional[str] = None,
    ) -> FieldApproval:
        """Update one field's state / value / note. Returns the new record.
        State transitions are not enforced — any state can move to any
        other (the UI surface is what enforces sane flow)."""
        with self._lock:
            fields = self._read(extraction_id)
            if field_key not in fields:
                raise KeyError(f"No such field {field_key!r} in extraction "
                               f"{extraction_id} — call initialise_from_extractions first.")
            rec = fields[field_key]
            if state is not None:
                rec.state = state
            if current_value is not None:
                rec.current_value = current_value
                # Auto-promote to USER_CORRECTED when the value changed
                # and the caller didn't explicitly set state to something else
                if state is None and current_value != rec.original_value:
                    rec.state = FieldState.USER_CORRECTED
            if user_note is not None:
                rec.user_note = user_note
            rec.updated_at = dt.datetime.now(dt.timezone.utc).isoformat()
            fields[field_key] = rec
            self._write(extraction_id, fields)
            return rec

    def confirm(self, extraction_id: str, field_key: str) -> FieldApproval:
        """One-shot helper: mark as user_confirmed (Sight + Accept)."""
        return self.update(extraction_id, field_key,
                           state=FieldState.USER_CONFIRMED)

    def reject(self, extraction_id: str, field_key: str,
                user_note: Optional[str] = None) -> FieldApproval:
        return self.update(extraction_id, field_key,
                           state=FieldState.REJECTED, user_note=user_note)

    def revert(self, extraction_id: str, field_key: str) -> FieldApproval:
        """Roll back to original AI value + ai_suggested state."""
        with self._lock:
            fields = self._read(extraction_id)
            if field_key not in fields:
                raise KeyError(field_key)
            rec = fields[field_key]
            rec.current_value = rec.original_value
            rec.state = FieldState.AI_SUGGESTED
            rec.user_note = None
            rec.updated_at = dt.datetime.now(dt.timezone.utc).isoformat()
            fields[field_key] = rec
            self._write(extraction_id, fields)
            return rec

    # ---------- gate logic ----------

    def is_ready_to_save(self, extraction_id: str) -> tuple[bool, list[str]]:
        """The proposal's Calculate/Save hard-block. Returns
        (ready, list_of_blocking_field_keys).

        Ready when every tracked field's state.is_resolved (confirmed,
        corrected, or rejected). Blocked while any field is still
        ai_suggested or pending.
        """
        with self._lock:
            fields = self._read(extraction_id)
            if not fields:
                return False, ["__no_extraction_yet__"]
            blockers = [k for k, f in fields.items() if not f.state.is_resolved]
            return (len(blockers) == 0, blockers)

    def progress(self, extraction_id: str) -> tuple[int, int]:
        """Return (resolved_count, total_count). For the progress bar."""
        with self._lock:
            fields = self._read(extraction_id)
            if not fields:
                return (0, 0)
            resolved = sum(1 for f in fields.values() if f.state.is_resolved)
            return (resolved, len(fields))

    def export_corrections(
        self, extraction_id: str,
    ) -> list[dict]:
        """For the admin-feedback page (Phase 6). List of fields where
        the user_corrected the AI's value, with both versions."""
        with self._lock:
            fields = self._read(extraction_id)
            return [
                {
                    "field_key": rec.field_key,
                    "ai_value": rec.original_value,
                    "user_value": rec.current_value,
                    "ai_confidence": rec.original_confidence,
                    "user_note": rec.user_note,
                    "updated_at": rec.updated_at,
                }
                for rec in fields.values()
                if rec.state == FieldState.USER_CORRECTED
            ]


# ---------- module-level default store ----------

_DEFAULT_STORE: Optional[ApprovalStore] = None


def get_default_store() -> ApprovalStore:
    """Lazy-init singleton. Streamlit reruns share this one instance
    in-process; persistence is via the JSON file on disk."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = ApprovalStore()
    return _DEFAULT_STORE
