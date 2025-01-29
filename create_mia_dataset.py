import os
import pdb
from tqdm import tqdm
import torch
import random
import numpy as np
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets
import argparse


def filter_data(data, min_length, max_length, args, domain):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    if domain == "code search net":
        key = "func_code_tokens"
    for i in tqdm(range(0, len(data[key]), args.batch_size)):
        batch = data[i:i + args.batch_size]
        texts = [item for item in batch]
        lengths = [len(text.split()) for text in texts]
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




parser = argparse.ArgumentParser()
parser.add_argument("--list", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--sample_size", type=int, default=1000)
parser.add_argument("--select_method", type=str, default="truncate", choices=["truncate", "nontruncate"])
parser.add_argument("--relative_length", type=str, default="False")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
seed_list = [10345, 19238, 19093]
#I will directly corpy "pile arxiv"
#data_list = ["code search net", "dolma wiki", "dolma stack", "m2d2"]
data_list = ["code search net"]
length_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, "rest"]
enumerate_length = len(length_list)
for idx, seed in enumerate(seed_list):
    for domain in data_list:
        if domain == "code search net":
            dataset = load_dataset("code-search-net/code_search_net")
            member_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            #merge valid and test
            non_member_dataset = concatenate_datasets([valid_dataset, test_dataset])
        for i in range(enumerate_length):
            if length_list[i] == 0:
                min_length = 5
                max_length = length_list[i + 1]
            elif length_list[i] == "rest":
                min_length = 1000
                max_length = 100000000000
            else:
                min_length = length_list[i]
                max_length = min_length + 100
            filtered_member_data = load_and_filter_data(member_dataset, min_length, max_length, args, domain)
            filtered_nonmember_data = load_and_filter_data(non_member_dataset, min_length, max_length, args, domain)
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
            os.makedirs(f"./dolma_absolute_filtered_dataset_{idx + 1}/{min_length}_{max_length}_truncated/{domain}",
                        exist_ok=True)
            dataset.save_to_disk(
                f"./dolma_absolute_filtered_dataset_{idx + 1}/{min_length}_{max_length}_truncated/{domain}")

