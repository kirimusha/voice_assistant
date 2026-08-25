from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sounddevice as sd


class WakeWordDetector:
    """wake-word детектор на базе OpenWakeWord и микрофона."""

    def __init__(
        self,
        wake_word_model: str = DEFAULT_WAKE_WORD_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE,
        threshold: float = DEFAULT_WAKE_WORD_THRESHOLD,
    ) -> None:
        self.wake_word_model = wake_word_model
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self._model = None
        self._stream = None
        self._model_label = None
        self._init_error = None
        self._audio_buffer = []
        self._load_model()

    def _load_model(self) -> None:
        try:
            import openwakeword
        except ImportError as exc:
            self._init_error = exc
            self._model = None
            self._model_label = None
            return

        local_model = ROOT_DIR / "models" / "openwakeword" / f"{self.wake_word_model}_v0.1.onnx"
        if local_model.exists():
            model_path = str(local_model)
            try:
                self._model = openwakeword.Model(
                    wakeword_models=[model_path],
                    inference_framework="onnx",
                )
                self._model_label = list(self._model.models.keys())[0]
                return
            except Exception as exc:
                self._init_error = exc
                self._model = None
                self._model_label = None
                return

        try:
            self._model = openwakeword.Model(
                wakeword_models=[self.wake_word_model],
                inference_framework="onnx",
            )
            self._model_label = list(self._model.models.keys())[0]
        except Exception as exc:
            self._init_error = exc
            self._model = None
            self._model_label = None

    def _normalize_audio(self, frames: np.ndarray) -> np.ndarray:
        audio = np.asarray(frames, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.reshape(-1)
        elif audio.ndim > 1:
            audio = audio.reshape(-1)
        return audio / np.iinfo(np.int16).max

    def wait_for_wake_word(self, timeout: float = 3.0) -> bool:
        if self._model is None or self._model_label is None:
            print("[WakeWord] OpenWakeWord модель недоступна, режим fallback активен.")
            return False

        deadline = time.time() + timeout
        min_frames = 3
        max_frames = 8
        self._audio_buffer = []

        try:
            if self._stream is None:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=self.block_size,
                    latency="low",
                )
                self._stream.start()
        except Exception as exc:
            print(f"[WakeWord] Не удалось открыть микрофон: {exc}")
            return False

        chunk_counter = 0
        while time.time() < deadline:
            try:
                frames, _ = self._stream.read(self.block_size)
            except Exception as exc:
                print(f"[WakeWord] Ошибка чтения аудио: {exc}")
                return False

            audio = self._normalize_audio(frames)
            self._audio_buffer.append(audio)
            if len(self._audio_buffer) > max_frames:
                self._audio_buffer = self._audio_buffer[-max_frames:]

            if len(self._audio_buffer) < min_frames:
                continue

            buffered_audio = np.concatenate(self._audio_buffer)
            try:
                scores = self._model.predict(
                    buffered_audio,
                    threshold={self._model_label: self.threshold},
                )
            except Exception as exc:
                print(f"[WakeWord] Ошибка предсказания: {exc}")
                return False

            score = scores.get(self._model_label, 0.0)
            chunk_counter += 1
            if chunk_counter % 5 == 0:
                print(f"[WakeWord] score={score:.3f} threshold={self.threshold:.3f}")

            if score >= self.threshold:
                return True

        return False
