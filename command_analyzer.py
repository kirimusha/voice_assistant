from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterable


class CommandAnalyzer:
    """Находит заданные команды в новых записях файла джисон"""

    def __init__(
        self,
        json_file: str | Path | None = None,
        commands: Iterable[str] = ("стоп",),
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        self.json_file = (
            Path(json_file)
            if json_file is not None
            else Path(__file__).with_name("transcription.json")
        )
        self.commands = tuple(dict.fromkeys(command.strip() for command in commands if command.strip()))
        self._on_stop = on_stop
        self._processed_records = 0

    def analyze_text(self, text: str) -> list[str]:
        """Находит команды в одном распознанном фрагменте текста."""
        found_commands: list[str] = []
        for command in self.commands:
            pattern = rf"(?<!\w){re.escape(command)}(?!\w)"
            if re.search(pattern, text, flags=re.IGNORECASE):
                found_commands.append(command)
                self._handle_command(command, text)
        return found_commands

    def analyze_new_records(self) -> list[str]:
        """Обрабатывает ещё не просмотренные записи и возвращает найденные команды."""
        try:
            with self.json_file.open("r", encoding="utf-8") as file:
                records = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

        if not isinstance(records, list):
            return []

        if len(records) < self._processed_records:
            self._processed_records = 0

        found_commands: list[str] = []
        for record in records[self._processed_records :]:
            if not isinstance(record, dict):
                continue

            text = record.get("text")
            if not isinstance(text, str):
                continue

            found_commands.extend(self.analyze_text(text))

        self._processed_records = len(records)
        return found_commands

    def _handle_command(self, command: str, text: str) -> None:
        if command.casefold() == "стоп" and self._on_stop is not None:
            self._on_stop()