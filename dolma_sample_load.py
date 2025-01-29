import numpy as np
import torch
from bokeh.server.django import directory
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pdb
import os

class MemmapTokenDataset(Dataset):
    def __init__(self, file_path, seq_len=128, dtype="uint32"):
        self.seq_len = seq_len
        self.memmap_data = np.memmap(file_path, dtype=dtype, mode="r")
        self.total_len = len(self.memmap_data)
        self.num_segments = self.total_len // seq_len

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        token_ids = self.memmap_data[start:end]
        token_ids_arr = token_ids.astype(np.int32, copy=True)
        token_ids_tensor = torch.from_numpy(token_ids_arr)
        return token_ids_tensor


def collate_fn(batch):
    return torch.stack(batch, dim=0)

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def save_text_dataset(text_dataset, filename, directory="saved_datasets"):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{filename}.pt")
    torch.save(text_dataset, file_path)
    print(f"Saved dataset to {file_path}")

def load_text_dataset(filename, directory="saved_datasets"):
    file_path = os.path.join(directory, f"{filename}.pt")
    text_dataset = torch.load(file_path)
    print(f"Loaded dataset from {file_path}")
    return text_dataset


if __name__ == "__main__":
    file_list = [
        "c4_en_valid.npy", "dolma_book_valid.npy", "m2d2_s2orc_valid.npy",
        "pile_valid.npy", "stack_valid.npy", "wikipedia_v103_valid.npy",
        "common_crawl_valid.npy", "ice_valid.npy", "pes2o_valid.npy",
        "reddit_valid.npy", "wiki_valid.npy"
    ]
    root_file = "data_OLMo2_13b_1124/eval_data/"

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
    seq_len_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]

    for file_name in file_list:
        file_path = os.path.join(root_file, file_name)
        base_name = os.path.splitext(file_name)[0]  # Extract file name without extension

        for seq_len in seq_len_list:
            dataset = MemmapTokenDataset(file_path=file_path, seq_len=seq_len, dtype="uint32")
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )

            samples = []
            for step, batch_token_ids in enumerate(dataloader):
                if len(samples) >= 150:
                    break
                text = tokenizer.decode(batch_token_ids[0].tolist(), skip_special_tokens=True)
                samples.append(text)

            # Store the samples in a dataset
            text_dataset = TextDataset(samples)

            # Save each dataset
            file_label = f"{base_name}_seq_len_{seq_len}"
            save_text_dataset(text_dataset, file_label, directory="data_OLMo2_13b_1124/eval_data/processed_data")


