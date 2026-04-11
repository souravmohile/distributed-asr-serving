from fastapi import FastAPI, UploadFile
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
from dotenv import load_dotenv

load_dotenv()

processor = AutoProcessor.from_pretrained("openai/whisper-base", cache_dir="./models/")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base", cache_dir="./models/")

app = FastAPI()


@app.get('/models')
def available_models():
    """Returns a list of available models for performing transcription."""

    models = ["openai/whisper-base"]

    return {
        "AvailableModels": models
        }


@app.post('/transcribe')
def transcribe(audio_file: UploadFile):
    """Splits the input audio_file into batches of 30s which are then transcribed using the whisper model. 
    These transcripts are then combindes and returned back to the user."""

    audio, sampling_rate = librosa.load(audio_file.file, sr=16000)

    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

    predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return {
        "filename": audio_file.filename,
        "transcript": transcription[0]
        }
