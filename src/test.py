from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")

audio, sampling_rate = librosa.load("data/audio_files/audio_file_1.mp3", sr=16000)

input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

# 4. Generate token IDs
predicted_ids = model.generate(input_features)

# 5. Decode token IDs to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])