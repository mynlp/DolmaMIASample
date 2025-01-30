import os
import pdb
from tqdm import tqdm
import torch
import random
import numpy as np
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets, load_from_disk
import argparse
from dolma_sample_load import MemmapTokenDataset, collate_fn
from torch.utils.data import Dataset, DataLoader


def filter_data(data, min_length, max_length, args, domain):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    if domain == "code search net":
        key = "func_code_tokens"
    elif domain in ["algebraic-stack", "open-web-math", "arxiv"]:
        key = "text"
    for i in tqdm(range(0, len(data[key]), args.batch_size)):
        batch = data[key][i:i + args.batch_size]
        texts = [item for item in batch]
        if domain == "code search net":
            lengths = [len(text) for text in texts]
        else:
            lengths = [len(text.split()) for text in texts]
        #pdb.set_trace()
        if args.select_method == "nontruncate":
            valid_indices = (np.array(lengths) >= min_length) & (np.array(lengths) <= max_length)
            filtered_data.extend([batch[j] for j in range(len(batch)) if valid_indices[j]])
        elif args.select_method == "truncate" and args.relative_length == "False":
            valid_indices = (np.array(lengths) >= min_length)
            filtered_data.extend([" ".join(batch[j].split()[:max_length]) for j in range(len(batch)) if valid_indices[j]])
    return filtered_data

def load_and_filter_data(dataset, min_length, max_length, args, domain):
    """filtering and load"""
    merged_data = []
    filtered_data = filter_data(dataset, min_length, max_length, args, domain)
    merged_data.extend(filtered_data)
    if len(merged_data) > args.sample_size:
        return random.sample(merged_data, args.sample_size)
    return merged_data

def load_and_filter_npy_data(dataset, args):
    """filtering and load"""
    merged_data = []
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    for step, batch_token_ids in enumerate(dataloader):
        text = tokenizer.decode(batch_token_ids[0].tolist(), skip_special_tokens=True)
        merged_data.append(text)
        if step % 1000 == 0:
            print(f"Processed {step} samples")
            if step == 50000:
                break
    if len(merged_data) > args.sample_size:
        return random.sample(merged_data, args.sample_size)
    return merged_data

def load_text_dataset(filename, directory="saved_datasets"):
    file_path = os.path.join(directory, f"{filename}.pt")
    text_dataset = torch.load(file_path)
    print(f"Loaded dataset from {file_path}")
    return text_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--list", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--select_method", type=str, default="truncate", choices=["truncate", "nontruncate"])
parser.add_argument("--relative_length", type=str, default="False")
parser.add_argument("--domain", type=str, default="arxiv")
parser.add_argument("--device", type=str, default="beyondai")
args = parser.parse_args()
if args.device == "wisteria":
    prefix = "."
elif args.device == "chomusuke":
    prefix = "data/bwchen"
elif args.device == "beyondai":
    prefix = "/store/Dolma"
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
seed_list = [10345, 19238, 19093]
#data_list = ["code search net", "dolma wiki", "dolma stack", "m2d2", "arxiv", "open-web-math", "algebraic-stack"]
data_list = ["arxiv"]
length_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, "rest"]
enumerate_length = len(length_list)
sample_num = 200000
for idx, seed in enumerate(seed_list):
    random.seed(seed)
    if args.domain == "code search net":
        dataset = load_dataset("code-search-net/code_search_net")
        member_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        #merge valid and test
        non_member_dataset = concatenate_datasets([valid_dataset, test_dataset])
    elif args.domain == "dolma wiki":
        member_dataset_path = "data_OLMo2_13b_1124/train_data/raw_data/wiki_train.npy"
        non_member_dataset_path = "data_OLMo2_13b_1124/eval_data/raw_data/wiki_valid.npy"
    elif args.domain == "dolma pes2o":
        member_dataset_path = "data_OLMo2_13b_1124/train_data/raw_data/pes2o_train.npy"
        non_member_dataset_path = "data_OLMo2_13b_1124/eval_data/raw_data/pes2o_valid.npy"
    elif args.domain == "algebraic-stack":
        if args.device == "wisteria":
            dataset = load_dataset("EleutherAI/proof-pile-2", "algebraic-stack")
        else:
            dataset = load_dataset("EleutherAI/proof-pile-2", "algebraic-stack", cache_dir=f"{prefix}")
        member_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        if os.path.exists(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}"):
            member_dataset = load_from_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
        else:
            random_indices = random.sample(range(len(member_dataset)),
                                           k=sample_num if sample_num < len(member_dataset) else len(
                                               member_dataset))
            member_dataset = member_dataset.select(random_indices)
            os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}", exist_ok=True)
            member_dataset.save_to_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
            member_dataset = load_from_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
        #merge valid and test
        validation_sampled = valid_dataset
        test_sampled = test_dataset
        non_member_dataset = concatenate_datasets([validation_sampled, test_sampled])
    elif args.domain == "arxiv":
        if args.device == "wisteria":
            dataset = load_dataset("EleutherAI/proof-pile-2", "arxiv")
        else:
            dataset = load_dataset("EleutherAI/proof-pile-2", "arxiv", cache_dir=f"{prefix}")
        member_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        if os.path.exists(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}"):
            member_dataset = load_from_disk(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
        else:
            random_indices = random.sample(range(len(member_dataset)),
                                           k=sample_num if sample_num < len(member_dataset) else len(
                                               member_dataset))
            member_dataset = member_dataset.select(random_indices)
            os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
                        exist_ok=True)
            member_dataset.save_to_disk(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
            member_dataset = load_from_disk(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
        # merge valid and test
        validation_sampled = valid_dataset
        test_sampled = test_dataset
        non_member_dataset = concatenate_datasets([validation_sampled, test_sampled])
    elif args.domain == "open-web-math":
        if args.device == "wisteria":
            dataset = load_dataset("EleutherAI/proof-pile-2", "open-web-math")
        else:
            dataset = load_dataset("EleutherAI/proof-pile-2", "open-web-math", cache_dir=f"{prefix}")
        member_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        if os.path.exists(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}"):
            member_dataset = load_from_disk(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
        else:
            random_indices = random.sample(range(len(member_dataset)),
                                           k=sample_num if sample_num < len(member_dataset) else len(
                                               member_dataset))
            member_dataset = member_dataset.select(random_indices)
            os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
                        exist_ok=True)
            member_dataset.save_to_disk(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
            member_dataset = load_from_disk(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
        # merge valid and test
        validation_sampled = valid_dataset
        test_sampled = test_dataset
        non_member_dataset = concatenate_datasets([validation_sampled, test_sampled])
    for i in range(enumerate_length):
        print (f"Processing {args.domain} with length {length_list[i]}")
        if length_list[i] == 0:
            min_length = 5
            max_length = length_list[i + 1]
        elif length_list[i] == "rest":
            continue
        else:
            min_length = length_list[i]
            max_length = min_length + 100
        if args.domain in ["dolma wiki", "dolma pes2o"]:
            member_dataset = MemmapTokenDataset(member_dataset_path, seq_len=max_length, dtype="uint32")
            non_member_dataset = MemmapTokenDataset(non_member_dataset_path, seq_len=max_length, dtype="uint32")
            filtered_member_data = load_and_filter_npy_data(member_dataset, args)
            filtered_nonmember_data = load_and_filter_npy_data(non_member_dataset, args)
        else:
            filtered_member_data = load_and_filter_data(member_dataset, min_length, max_length, args, args.domain)
            filtered_nonmember_data = load_and_filter_data(non_member_dataset, min_length, max_length, args, args.domain)
        member_data = []
        nonmember_data = []
        member_data.extend(filtered_member_data)
        nonmember_data.extend(filtered_nonmember_data)
        train_dataset = Dataset.from_dict({"data": member_data})
        test_dataset_short = Dataset.from_dict({"data": nonmember_data})
        dataset = DatasetDict({
            'member': train_dataset,
            'nonmember': test_dataset_short,
        })
        os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{min_length}_{max_length}_truncated/{args.domain}",
                    exist_ok=True)
        dataset.save_to_disk(
            f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{min_length}_{max_length}_truncated/{args.domain}")

