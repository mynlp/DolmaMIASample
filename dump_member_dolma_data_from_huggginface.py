from datasets import load_dataset


dataset = load_dataset("allenai/olmo-mix-1124", "pes2o", split="train", streaming=True)
print(dataset)