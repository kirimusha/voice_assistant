from __future__ import annotations

from recognizer import ASRRecognizer


class VoiceAssistant:
    def __init__(self) -> None:
        self.asr = TextRecognizer()

    def run(self) -> None:
        print("Голосовой ассистент запущен.")
        self.asr.listen_forever()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()