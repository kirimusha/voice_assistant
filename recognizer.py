from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model


class TextRecognizer:
    """Распознавание речи через локальный Vosk ASR"""
    def __init__(
        self,
        model_path: str = "/Users/kirimusha/projects/voice_assistant/models/vosk-model-small-ru-0.22",
        json_file: str = "/Users/kirimusha/projects/voice_assistant/transcription.json",
        sample_rate: int = 16000,
        block_size: int = 2000,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.json_file = json_file

        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

        self._last_text: str = ""

    def _save_to_json(self, text: str) -> None:
        """Сохраняет одну запись в JSON файл."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
        }

        if os.path.exists(self.json_file):
            with open(self.json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(entry)

        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(status)

        if self.recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")
            if text:
                self._last_text = text
                self._save_to_json(text)
            print("Распознано:", text)
        else:
            partial = json.loads(self.recognizer.PartialResult())
            print("Промежуточно:", partial.get("partial", ""))

    def listen_forever(self) -> None:
        """Запускает бесконечное прослушивание микрофона (как в workwithvosk.py)."""
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            print("Начало записи, говорите...")
            while True:
                sd.sleep(1000)