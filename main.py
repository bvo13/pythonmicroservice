import httpx
import tempfile
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from pydub import AudioSegment
import os
from voice_emotion_classification import EmotionClassificationPipeline
from google.cloud import speech
import asyncio
import re
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from torch import nn
from concurrent.futures import ThreadPoolExecutor


thread_pool = ThreadPoolExecutor(max_workers=4)



feature_extractor = None
wav2vec_model = None
classifier = None
thresholds = None

@app.on_event("startup")
async def startup_event():
    global feature_extractor, wav2vec_model, classifier, thresholds
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(DEFAULT_MODEL_NAME)
    wav2vec_model = Wav2Vec2Model.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    wav2vec_model.eval()
    
   
    classifier_path = "C:/Users/brady/training/cache_sep28k/runs/best_model.pt"
    thresholds_path = "C:/Users/brady/training/cache_sep28k/runs/thresholds.npy"
    
    checkpoint = torch.load(classifier_path, map_location=device)
    classifier = MLPClassifier(
        in_dim=checkpoint['in_dim'],
        num_labels=len(LABEL_COLUMNS),
        hidden=checkpoint['cfg']['hidden'],
        dropout=checkpoint['cfg']['dropout']
    )
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier.eval()
    
    thresholds = np.load(thresholds_path)


@app.get("/")
def root():
    return "hello"

class URL(BaseModel):
    url: str

LABEL_COLUMNS = [
    "Unsure", "PoorAudioQuality", "Prolongation", "Block", "SoundRep",
    "WordRep", "DifficultToUnderstand", "Interjection", "NoStutteredWords",
    "NaturalPause", "Music", "NoSpeech",
]

TARGET_SAMPLE_RATE = 16000
DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"

class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, hidden: int = 384, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.net(x)

def load_audio(audio_path: str, target_sr: int = TARGET_SAMPLE_RATE):
    """Load and preprocess audio file."""
    wav_np, sr = sf.read(str(audio_path))
    wav = torch.from_numpy(wav_np).float()
    
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    
    if sr != target_sr:
        from torchaudio.transforms import Resample
        resampler = Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    
    return wav.squeeze(0)  # Return (T,)

def extract_embedding(
    audio: torch.Tensor,
    feature_extractor: Wav2Vec2FeatureExtractor,
    wav2vec_model: Wav2Vec2Model,
    device: str
) -> torch.Tensor:
    """Extract wav2vec embedding from audio."""
    with torch.no_grad():
        
        inputs = feature_extractor(
            audio.numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
       
        outputs = wav2vec_model(**inputs)
        hidden = outputs.last_hidden_state
        
      
        embedding = hidden.mean(dim=1)
        
    return embedding.cpu()

def predict_audio_file(
    audio_path: str,
    classifier_path: str,
    thresholds_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """End-to-end prediction pipeline for a single audio file."""
    
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(DEFAULT_MODEL_NAME)
    wav2vec_model = Wav2Vec2Model.from_pretrained(DEFAULT_MODEL_NAME).to(device)
    wav2vec_model.eval()
    
    
    checkpoint = torch.load(classifier_path, map_location=device)
    classifier = MLPClassifier(
        in_dim=checkpoint['in_dim'],
        num_labels=len(LABEL_COLUMNS),
        hidden=checkpoint['cfg']['hidden'],
        dropout=checkpoint['cfg']['dropout']
    )
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.to(device)
    classifier.eval()
    
   
    thresholds = np.load(thresholds_path) if thresholds_path else np.array([0.5] * len(LABEL_COLUMNS))
    
 
    audio = load_audio(audio_path)
    embedding = extract_embedding(audio, feature_extractor, wav2vec_model, device)
    
   
    with torch.no_grad():
        embedding = embedding.to(device)
        logits = classifier(embedding)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= thresholds).astype(int)
    
   
    results = {
        "predictions": {label: bool(pred) for label, pred in zip(LABEL_COLUMNS, preds[0])},
        "probabilities": {label: float(prob) for label, prob in zip(LABEL_COLUMNS, probs[0])}
    }
    
    return results


if __name__ == "__main__":
    CLASSIFIER_PATH = "cache_sep28k/runs/best_model.pt"
    THRESHOLDS_PATH = "cache_sep28k/runs/thresholds.npy"
    
    audio_path = "path/to/your/audio.wav"
    results = predict_audio_file(
        audio_path=audio_path,
        classifier_path=CLASSIFIER_PATH,
        thresholds_path=THRESHOLDS_PATH
    )
    
    print("\nPredictions:")
    for label, is_present in results["predictions"].items():
        if is_present:
            prob = results["probabilities"][label]
            print(f"{label}: {prob:.3f}")


@app.post("/analyze")
async def download_and_handle_audio(item: URL):
    timeout_settings = httpx.Timeout(timeout=60.0) 
    print(item.url)
    async with httpx.AsyncClient(timeout=timeout_settings) as client:
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
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted_file:
                converted_audio.export(converted_file.name, format = "wav", parameters=["-acodec", "pcm_s16le"])
                converted_file.flush()
                temp_converted_audio_path = converted_file.name
            
           
            
            
            

            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            try:
                def run_inference():
                    audio = load_audio(temp_converted_audio_path)
                    embedding = extract_embedding(audio, feature_extractor, wav2vec_model, device)
                    with torch.no_grad():
                        embedding = embedding.to(device)
                        logits = classifier(embedding)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        preds = (probs >= thresholds).astype(int)
                    return preds, probs
                
                loop = asyncio.get_event_loop()
                preds, probs = await loop.run_in_executor(thread_pool, run_inference)
                
                
                true_predictions = []
                false_predictions = []
                
                for label, pred, prob in zip(LABEL_COLUMNS, preds[0], probs[0]):
                    if pred == 1:
                        true_predictions.append(f"{label} ({prob:.3f})")
                    else:
                        false_predictions.append(f"{label} ({prob:.3f})")
                
                
                predictions_str = "Displays the following traits:\n"
                if true_predictions:
                    predictions_str += "- " + "\n- ".join(true_predictions)
                else:
                    predictions_str += "None"
                
                predictions_str = str(predictions_str).strip()
                
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                return JSONResponse(content={"error": str(e)}, status_code=500)
            
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
            
            response_data = {
                "traits": predictions_str,
                "transcription": audio_transcription,
                "fillerWords": detected_fillers,
                "wpm": words_per_minute
            }
            
            
            
            return JSONResponse(
                content=response_data,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
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