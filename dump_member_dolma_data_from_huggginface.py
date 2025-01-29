from datasets import load_dataset

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

# Use the batched_data function to iterate over the dataset in batches
for batch in batched_data(dataset, batch_size):
    print(batch)