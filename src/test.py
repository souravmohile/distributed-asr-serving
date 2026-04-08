from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")

