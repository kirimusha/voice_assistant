from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model


class ASRRecognizer:
    """Реальное распознавание речи через локальный Vosk ASR."""

    def __init__(self, use_demo_input: bool = False) -> None:
        self.use_demo_input = use_demo_input
        self.vosk_model_dir = VOSK_MODEL_DIR
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.block_size = 4000
        self.input_device_index = self._resolve_input_device()
        self._model = None
        self._recognizer = None
        self._load_model()

    def _resolve_input_device(self) -> int | None:
        try:
            devices = sd.query_devices()
        except Exception:
            return None

        if isinstance(devices, list):
            for idx, device in enumerate(devices):
                name = str(device.get("name", "")).lower()
                if "microphone" in name and device.get("max_input_channels", 0) > 0:
                    print(f"[ASR] selected microphone device: {idx} -> {device.get('name')}")
                    return idx

        return None

    def _load_model(self) -> None:
        model_dir = self._resolve_model_dir()
        if model_dir is None:
            print(
                f"[ASR] Vosk model directory not found. "
                f"Put a model into {self.vosk_model_dir} and restart the assistant."
            )
            return

        try:
            self._model = Model(model_path=str(model_dir))
            self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
            self._recognizer.SetWords(False)
            self._recognizer.SetPartialWords(True)
            print(f"[ASR] Vosk model loaded from: {model_dir}")
        except Exception as exc:
            print(f"[ASR] Не удалось загрузить Vosk-модель: {exc}")
            self._model = None
            self._recognizer = None

    def _resolve_model_dir(self) -> Path | None:
        if not self.vosk_model_dir.exists():
            return None

        candidates = [
            self.vosk_model_dir,
            self.vosk_model_dir / "model",
        ]

        for candidate in candidates:
            if candidate.exists() and (
                (candidate / "conf").exists() or (candidate / "am").exists()
            ):
                return candidate

        for child in sorted(self.vosk_model_dir.iterdir()):
            if child.is_dir() and (
                (child / "conf").exists() or (child / "am").exists()
            ):
                return child

        return None

    def _read_live_transcript(self, timeout: float = 3.0) -> str:
        if self._recognizer is None or self._model is None:
            return ""

        deadline = time.monotonic() + timeout
        transcript = ""

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.block_size,
                latency="low",
                device=self.input_device_index,
            ) as stream:
                chunk_index = 0
                while time.monotonic() < deadline:
                    frames, _ = stream.read(self.block_size)
                    audio = np.asarray(frames, dtype=np.int16)
                    if audio.ndim == 2:
                        audio = audio.reshape(-1)

                    if audio.size == 0:
                        continue

                    audio_energy = float(np.sqrt(np.mean(np.abs(audio.astype(np.float32)) ** 2)))
                    chunk_index += 1
                    if chunk_index % 5 == 0:
                        print(f"[ASR] mic_level={audio_energy:.3f}")

                    if self._recognizer.AcceptWaveform(audio.tobytes()):
                        result = json.loads(self._recognizer.Result())
                        candidate = result.get("text", "").strip()
                        if candidate:
                            transcript = candidate
                            print(f"[ASR] final={transcript}")
                            return transcript

                    partial = json.loads(self._recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text:
                        print(f"[ASR] partial={partial_text}")

                final_result = json.loads(self._recognizer.FinalResult())
                transcript = final_result.get("text", "").strip()
                if transcript:
                    print(f"[ASR] final={transcript}")
                return transcript
        except Exception as exc:
            print(f"[ASR] Ошибка записи/распознавания микрофона: {exc}")
            return ""

    def listen_and_recognize(self, timeout: float = 3.0) -> str:
        try:
            if self.use_demo_input:
                return input("Введите команду: ").strip()

            transcript = self._read_live_transcript(timeout=timeout)
            if transcript:
                return transcript

            return input("Введите команду (Vosk не распознал аудио): ").strip()
        except EOFError:
            return ""
