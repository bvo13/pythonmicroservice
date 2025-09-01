import httpx
import tempfile
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from transformers import pipeline
from pydub import AudioSegment
import os
from voice_emotion_classification import EmotionClassificationPipeline
from google.cloud import speech
import asyncio
import re


app = FastAPI()

classifier = EmotionClassificationPipeline.from_pretrained("griko/emotion_8_cls_svm_ecapa_ravdess")

@app.get("/")
def root():
    return "hello"

class URL(BaseModel):
    url: str


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
            converted_audio = converted_audio.set_channels(1).set_frame_rate(16000)
            duration_seconds = len(converted_audio) / 1000.0
            duration_minutes = duration_seconds / 60
            print(f"Audio Duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted_file:
                converted_audio.export(converted_file.name, format = "wav", parameters=["-acodec", "pcm_s16le"])
                converted_file.flush()
                temp_converted_audio_path = converted_file.name
            result = classifier(temp_converted_audio_path)
            audio_transcription = await transcribe_audio(temp_converted_audio_path)
            transcript = [word.strip().lower() for word in re.split(r'\W+', audio_transcription) if word.strip()]
            word_count = len(transcript)
            filler_variants = {
            "um": ["um", "umm", "ummm"],
            "uh": ["uh", "uhh", "uhhh"],
            "like": ["like"],
            "so": ["so"]
            }
            detected_fillers = sum(sum(transcript.count(variant) for variant in variants) for variants in filler_variants.values())
            words_per_minute = word_count / duration_minutes
            if duration_minutes > 0 and len(transcript) > 0:
                words_per_minute = word_count / duration_minutes
            return JSONResponse(content={
                "emotion": result[0],
                "transcription": audio_transcription,
                "fillerWords": detected_fillers,
                "wpm": words_per_minute
            })
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_converted_audio_path):
                os.remove(temp_converted_audio_path)

async def transcribe_audio(file_path: str) -> str:
    client = speech.SpeechClient()
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
    
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    response = client.long_running_recognize(config=config, audio=audio)
    result = response.result(timeout=600)  
    if not result.results:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No speech detected")
    full_transcript = " ".join([res.alternatives[0].transcript for res in result.results])

    return full_transcript