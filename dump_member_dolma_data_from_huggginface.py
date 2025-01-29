from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

def batched_data(dataset, batch_size):
    batch = []
    for sample in dataset:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

dataset = load_dataset("allenai/olmo-mix-1124", "pes2o", split="train", streaming=True)
print(dataset)

batch_size = 32

file_path = '/work/gk77/share/DolmaMIASample/train_data/raw_data/pes2o.npy'
estimated_total_samples = 100000
fp = np.memmap(file_path, dtype='uint32', mode='w+', shape=(estimated_total_samples, 128))
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
seq_len = 128
offset = 0
for batch in batched_data(dataset, batch_size):
    texts = [x["text"] for x in batch]
    encoded_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=seq_len,
                               return_tensors='np')
    batch_size_actual = len(encoded_inputs['input_ids'])
    fp[offset:offset + batch_size_actual] = encoded_inputs['input_ids']
    offset += batch_size_actual

fp.flush()
