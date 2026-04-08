import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
from dotenv import load_dotenv

load_dotenv()

processor = AutoProcessor.from_pretrained("openai/whisper-base", cache_dir="./models/")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base", cache_dir="./models/")

audio, sampling_rate = librosa.load("data/audio_files/audio_file_1.mp3", sr=16000)

input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

predicted_ids = model.generate(input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])