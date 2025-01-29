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

dataset = load_dataset("allenai/olmo-mix-1124", "pes2o", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
file_path = '/work/gk77/share/DolmaMIASample/data_OLMo2_13b_1124/train_data/raw_data/pes2o.npy'
seq_len = 128
batch_size = 32

# Initialize a list for holding batches
all_encoded_inputs = []

# Process each batch and add it to the list
for batch in batched_data(dataset, batch_size):
    texts = [x["text"] for x in batch]
    encoded_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=seq_len, return_tensors='pt')
    # Convert tensors to plain numpy array for easier handling
    input_ids_np = encoded_inputs['input_ids'].numpy()
    all_encoded_inputs.append(input_ids_np)

# Concatenate all batches to form a single numpy array
all_encodings = np.concatenate(all_encoded_inputs, axis=0)

# Write to memmap
# Create memmap with actual size
actual_total_samples = all_encodings.shape[0]
fp = np.memmap(file_path, dtype='uint32', mode='w+', shape=(actual_total_samples, seq_len))

# Copy data to file
fp[:] = all_encodings[:]
fp.flush()
