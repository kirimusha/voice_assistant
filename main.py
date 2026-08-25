from __future__ import annotations

from recognizer import ASRRecognizer


class VoiceAssistant:
    """Голосовой ассистент: непрерывное распознавание речи через Vosk."""

    def __init__(self) -> None:
        self.asr = ASRRecognizer()

    def run(self) -> None:
        print("Голосовой ассистент запущен.")
        self.asr.listen_forever()


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()