import httpx
import tempfile
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import pipeline
from pydub import AudioSegment
import os
from voice_emotion_classification import EmotionClassificationPipeline

app = FastAPI()

classifier = EmotionClassificationPipeline.from_pretrained("griko/emotion_8_cls_svm_ecapa_ravdess")

@app.get("/")
def root():
    return "hello"

@app.post("/analyze")
async def download_and_handle_audio(item: URL):
    print(item.url)
    async with httpx.AsyncClient() as client:
        response = await client.get(item.url)
        response.raise_for_status()
        audio = response.content
        content_type = response.headers.get("Content-Type","")
        if "mpeg" in content_type:
            ext = ".mp3"
        elif "wav" in content_type:
            ext = ".wav"
        elif "mp4" in content_type or "m4a" in content_type:
            ext = ".m4a"
        else:
            ext = ".mp3"
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temporary:
            temporary.write(audio)
            temporary.flush()
            temp_file_path = temporary.name
        try:
            converted_audio = AudioSegment.from_file(temp_file_path, format = ext.lstrip('.'))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted_file:
                converted_audio.export(converted_file.name, format = "wav")
                converted_file.flush()
                temp_converted_audio_path = converted_file.name

            result = classifier(converted_file.name)
            return JSONResponse(result)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_converted_audio_path):
                os.remove(temp_converted_audio_path)