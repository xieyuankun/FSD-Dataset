import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from datasets import load_dataset

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

item = ds[0]['audio']['array']
print(item)
print(item.shape)
input_values = processor(item, return_tensors="pt").input_values  # Batch size 1
print(input_values)
print(input_values.shape)
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print(transcription)