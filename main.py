from __future__ import annotations

from recognizer import ASRRecognizer
from detector import WakeWordDetector


class VoiceAssistant:
    """wake word и распознавание речи."""

    def __init__(self) -> None:
        self.wake_word_detector = WakeWordDetector()
        self.asr = ASRRecognizer()

    def run(self) -> None:
        print("Голосовой ассистент запущен. Жду wake word...")

        while True:
            detected = self.wake_word_detector.wait_for_wake_word(timeout=DEFAULT_TIMEOUT_SECONDS)
            if not detected:
                continue

            print("Wake word найден.")
            text = self.asr.listen_and_recognize(timeout=DEFAULT_TIMEOUT_SECONDS)
            if not text:
                continue

            print(f"Распознано: {text}")


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
