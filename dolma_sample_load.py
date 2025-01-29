import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pdb

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


if __name__ == "__main__":
    file_path = "data_OLMo2_13b_1124/eval_data/c4_en_valid.npy"
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")

    seq_len_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]

    # Dictionary to store results
    results = {}

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
            text = tokenizer.decode(batch_token_ids[0].tolist(), skip_special_tokens=True)
            samples.append(text)
        results[seq_len] = samples
        pdb.set_trace()

