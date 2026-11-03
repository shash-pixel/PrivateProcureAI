"""Local JSON persistence for preference memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
from uuid import UUID

from privateprocureai.schemas import UserPreferenceMemory


class PreferenceMemoryRepository(Protocol):
    """Storage contract replaceable by PostgreSQL in Module 18."""

    def get(self, profile_id: UUID) -> UserPreferenceMemory | None:
        """Load one profile's memory, if it exists."""

    def save(self, memory: UserPreferenceMemory) -> None:
        """Persist one profile's complete preference memory."""


class JsonPreferenceMemoryRepository:
    """File-backed local preference memory for development and demonstration."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def get(self, profile_id: UUID) -> UserPreferenceMemory | None:
        """Load a profile from the local JSON file."""
        records = self._read_records()
        payload = records.get(str(profile_id))

        if payload is None:
            return None

        return UserPreferenceMemory.model_validate(payload)

    def save(self, memory: UserPreferenceMemory) -> None:
        """Atomically save one profile without exposing data to source control."""
        records = self._read_records()
        records[str(memory.profile_id)] = memory.model_dump(mode="json")

        self._path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = self._path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self._path)

    def _read_records(self) -> dict[str, object]:
        """Read stored records or return an empty collection on first use."""
        if not self._path.exists():
            return {}

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Preference memory file is not valid JSON: {self._path}"
            ) from error

        if not isinstance(payload, dict):
            raise ValueError("Preference memory root must be a JSON object.")

        return payload