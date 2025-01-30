import os
import pdb
from tqdm import tqdm
import torch
import random
import numpy as np
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset, load_dataset, concatenate_datasets, load_from_disk
import argparse


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

def load_and_filter_npy_data(dataset, min_length, max_length, args, domain):
    """filtering and load"""
    merged_data = []
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
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
seed_list = [10345, 19238, 19093]
#I will directly corpy "pile arxiv"
#data_list = ["code search net", "dolma wiki", "dolma stack", "m2d2", "arxiv", "open-web-math", "algebraic-stack"]
data_list = ["algebraic-stack"]
length_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, "rest"]
enumerate_length = len(length_list)
sample_num = 200000
for idx, seed in enumerate(seed_list):
    #fix the random seed
    random.seed(seed)
    for domain in data_list:
        if domain == "code search net":
            dataset = load_dataset("code-search-net/code_search_net")
            member_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            #merge valid and test
            non_member_dataset = concatenate_datasets([valid_dataset, test_dataset])
        elif domain == "dolma wiki":
            member_dataset = load_text_dataset( f"wiki_train_seq_len_100", f"data_OLMo2_13b_1124/train_data/processed_data")
            non_member_dataset = load_text_dataset( f"wiki_valid_seq_len_200", f"data_OLMo2_13b_1124/eval_data/processed_data")
        elif domain == "algebraic-stack":
            dataset = load_dataset("EleutherAI/proof-pile-2", "algebraic-stack")
            member_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            prefix = "."
            if os.path.exists(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{domain}/raw_data/{seed}"):
                member_dataset = load_from_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{domain}/raw_data/{seed}")
            else:
                random_indices = random.sample(range(len(member_dataset)),
                                               k=sample_num if sample_num < len(member_dataset) else len(
                                                   member_dataset))
                member_dataset = member_dataset.select(random_indices)
                os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{domain}/raw_data/{seed}", exist_ok=True)
                member_dataset.save_to_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{domain}/raw_data/{seed}")
                member_dataset = load_from_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{domain}/raw_data/{seed}")
            #merge valid and test
            validation_sampled = valid_dataset
            test_sampled = test_dataset
            non_member_dataset = concatenate_datasets([validation_sampled, test_sampled])
        elif domain == "arxiv":
            dataset = load_dataset("EleutherAI/proof-pile-2", "arxiv")
            member_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            member_dataset = member_dataset.shuffle(seed=seed).select(range(min(sample_num, len(member_dataset))))
            validation_sampled = valid_dataset.shuffle(seed=seed).select(range(min(sample_num, len(valid_dataset))))
            test_sampled = test_dataset.shuffle(seed=seed).select(range(min(sample_num, len(test_dataset))))
            #merge valid and test
            non_member_dataset = concatenate_datasets([validation_sampled, test_sampled])
        elif domain == "open-web-math":
            dataset = load_dataset("EleutherAI/proof-pile-2", "open-web-math")
            member_dataset = dataset["train"]
            valid_dataset = dataset["validation"]
            test_dataset = dataset["test"]
            member_dataset = member_dataset.shuffle(seed=seed).select(range(min(sample_num, len(member_dataset))))
            validation_sampled = valid_dataset.shuffle(seed=seed).select(range(min(sample_num, len(valid_dataset))))
            test_sampled = test_dataset.shuffle(seed=seed).select(range(min(sample_num, len(test_dataset))))
            #merge valid and test
            non_member_dataset = concatenate_datasets([validation_sampled, test_sampled])
        for i in range(enumerate_length):
            print (f"Processing {domain} with length {length_list[i]}")
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

