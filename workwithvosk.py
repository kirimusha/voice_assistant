from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json
from datetime import datetime
import os

model = Model("/Users/kirimusha/projects/voice_assistant/models/vosk-model-small-ru-0.22")
rec = KaldiRecognizer(model, 16000)

json_file = "/Users/kirimusha/projects/voice_assistant/transcription.json"

def save_to_json(text):
    """Сохраняет одну запись в JSON файл"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text
    }
    
    # Проверяем, существует ли файл
    if os.path.exists(json_file):
        # Читаем существующие данные
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    data.append(entry)
    
    # Записываем обратно
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def callback(indata, frames, time, status):
    if status:
        print(status)
    # Для RawInputStream преобразуем напрямую
    if rec.AcceptWaveform(bytes(indata)):
        result = json.loads(rec.Result())
        text = result.get("text", "")
        save_to_json(text)
        print("Распознано:", text, "")
    else:
        partial = json.loads(rec.PartialResult())
        print("Промежуточно:", partial.get("partial", ""))

with sd.RawInputStream(samplerate=16000, blocksize=2000, dtype='int16',
                       channels=1, callback=callback):
    print("Начало записи, говорите...")
    while True:
        sd.sleep(1000)
