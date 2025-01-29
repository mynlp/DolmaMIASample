import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class MemmapTokenDataset(Dataset):
    def __init__(self, file_path, seq_len=128, dtype="uint32"):
        """
        Args:
            file_path (str): memmap 文件路径
            seq_len (int): 每次取多少个 token 进行解码
            dtype (str): memmap 对应的 numpy dtype (这里是 'uint32')
        """
        self.seq_len = seq_len
        self.memmap_data = np.memmap(file_path, dtype=dtype, mode="r")
        self.total_len = len(self.memmap_data)
        self.num_segments = self.total_len // seq_len  # 只取能整除的部分

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        # 从 memmap 取出一段 [seq_len] 的数据 (uint32)
        token_ids = self.memmap_data[start:end]
        # 将其拷贝到一个新的 numpy 数组，并转换为 int32（PyTorch 支持）
        token_ids_arr = token_ids.astype(np.int32, copy=True)  # copy=True生成可写数组
        # 再转为 torch.Tensor
        token_ids_tensor = torch.from_numpy(token_ids_arr)  # dtype=int32
        return token_ids_tensor

def collate_fn(batch):
    # batch 是一个列表，每个元素是 shape=[seq_len] 的 int32 Tensor
    # 这里使用 PyTorch 默认的堆叠方式即可
    return torch.stack(batch, dim=0)  # [batch_size, seq_len]

if __name__ == "__main__":
    file_list = ["c4_en_valid.npy", "dolma_book_valid.npy", "m2d2_s2orc_valid.npy", "pile_valid.npy",
                 "stack_valid.npy", "wikipedia_v103_valid.npy", "common_crawl_valid.npy", "ice_valid.npy",
                 "pes2o_valid.npy", "reddit_valid.npy", "wiki_valid.npy"]
    root_file = "data_OLMo2_13b_1124/eval_data/"
    file_path = root_file + file_list[0]
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
    dataset = MemmapTokenDataset(file_path=file_path, seq_len=128, dtype="uint32")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    for step, batch_token_ids in enumerate(dataloader):
        print("Step:", step, "batch shape:", batch_token_ids.shape, "dtype:", batch_token_ids.dtype)
        # 做一些处理...
        if step >= 1:
            break
        print(tokenizer.decode(batch_token_ids[0].tolist()))