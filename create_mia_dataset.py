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
from torch.utils.data import DataLoader
import gc
import torch
from torch.nn.utils.rnn import pad_sequence
import time
import pickle


#
def filter_data(data, min_length, max_length, args, domain):
    """批量过滤文本长度在给定Token数量范围的数据"""
    filtered_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if domain == "code_search_net":
        key = "func_code_tokens"
    elif domain in ["algebraic-stack", "open-web-math", "arxiv"]:
        key = "text"
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch_start_time = time.perf_counter()
        t0 = time.perf_counter()
        batch = [x[key] for x in data[i:i + args.batch_size]]
        batch_read_time = time.perf_counter() - t0
        t2 = time.perf_counter()
        if domain == "code_search_net":
            # Here items are assumed to be token lists so we just count their length.
            lengths = [len(item) for item in batch]
        else:
            # Tokenize each string once; we save the result so that we do splitting only one time per item.
            tokenized_batch = [text.split() for text in batch]
            lengths = [len(tokens) for tokens in tokenized_batch]
        batch_length_time = time.perf_counter() - t2
        #pdb.set_trace()
        t4 = time.perf_counter()
        lengths = torch.tensor(lengths, device=device)
        if args.select_method == "nontruncated":
            # Retain items only if their token count is in the desired range.
            indicator = lengths >= min_length and lengths <= max_length
            if domain == "code_search_net":
                filtered_data.extend(
                    [item for item, l in zip(batch, indicator) if l == True]
                )
            else:
                filtered_data.extend(
                    [" ".join(tokens) for tokens, l in zip(tokenized_batch, indicator)
                     if l == True]
                )

        elif args.select_method == "truncated" and args.relative_length == "False":
            # Here we drop items that do not reach the minimum length.
            indicator = lengths >= min_length
            if domain == "code_search_net":
                filtered_data.extend(
                    [" ".join(item[:max_length]) for item, l in zip(batch, indicator) if l == True]
                )
            else:
                filtered_data.extend(
                    [" ".join(tokens[:max_length]) for tokens, l in zip(tokenized_batch, indicator) if l == True]
                )
            # Remove or delay gc.collect() if not strictly necessary.
        batch_filtering_time = time.perf_counter() - t4
        batch_total_time = time.perf_counter() - batch_start_time
        print("  数据读取耗时:       {:.6f} 秒".format(batch_read_time))
        print("  Token长度计算耗时:   {:.6f} 秒".format(batch_length_time))
        print("  数据过滤处理耗时:    {:.6f} 秒".format(batch_filtering_time))
        print("  当前batch总耗时:      {:.6f} 秒".format(batch_total_time))
        print("-" * 40)
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
parser.add_argument("--dataset_idx", type=int, default=1)
args = parser.parse_args()
if args.device == "wisteria":
    prefix = "."
elif args.device == "chomusuke1":
    prefix = "/NAS/Personal/bwchen/Dolmad_ata"
elif args.device == "chomusuke2":
    prefix = "/store/bwchen"
elif args.device == "beyondai":
    prefix = "/store/Dolma"
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B")
#seed_list = [[0, 10345], [1, 19238], [2, 19093]]
if args.dataset_idx == 1:
    seed_list = [[0, 10345]]
elif args.dataset_idx == 2:
    seed_list = [[1, 19238]]
elif args.dataset_idx == 3:
    seed_list = [[2, 19093]]

#data_list = ["code search net", "dolma wiki", "dolma stack", "m2d2", "arxiv", "open-web-math", "algebraic-stack"]
data_list = ["arxiv"]
length_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, "rest"]
enumerate_length = len(length_list)
sample_num = 200000
for x in seed_list:
    seed = x[1]
    idx = x[0]
    random.seed(seed)
    if args.domain == "code_search_net":
        dataset = load_dataset("code-search-net/code_search_net")
        member_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        non_member_dataset = concatenate_datasets([valid_dataset, test_dataset])
        if os.path.exists(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl"):
            # member_dataset = load_from_disk(
            #     f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
            #     keep_in_memory=True
            # )
            member_dataset = pickle.load(open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl", "rb"))
            non_member_dataset = pickle.load(open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_valid.pkl", "rb"))
        else:
            random_indices = random.sample(range(len(member_dataset)),
                                           k=sample_num if sample_num < len(member_dataset) else len(
                                               member_dataset))
            member_dataset = list(member_dataset.select(random_indices))
            non_member_dataset = list(non_member_dataset)
            os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
                        exist_ok=True)
            pickle.dump(member_dataset, open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl", "wb"))
            pickle.dump(non_member_dataset, open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_valid.pkl", "wb"))
            #dump pickle member dataset
            # merge valid and test
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
        non_member_dataset = concatenate_datasets([valid_dataset, test_dataset])
        if os.path.exists(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl"):
            #member_dataset = load_from_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
            member_dataset = pickle.load(
                open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl",
                     "rb"))
            non_member_dataset = pickle.load(
                open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_valid.pkl",
                     "rb"))
        else:
            random_indices = random.sample(range(len(member_dataset)),
                                           k=sample_num if sample_num < len(member_dataset) else len(
                                               member_dataset))
            member_dataset = list(member_dataset.select(random_indices))
            non_member_dataset = list(non_member_dataset)
            os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
                        exist_ok=True)
            pickle.dump(member_dataset, open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl", "wb"))
            pickle.dump(non_member_dataset, open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_valid.pkl", "wb"))
            # random_indices = random.sample(range(len(member_dataset)),
            #                                k=sample_num if sample_num < len(member_dataset) else len(
            #                                    member_dataset))
            # member_dataset = member_dataset.select(random_indices)
            # os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}", exist_ok=True)
            # member_dataset.save_to_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
            # member_dataset = load_from_disk(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
        #merge valid and test
        #validation_sampled = valid_dataset
        #test_sampled = test_dataset
    elif args.domain == "arxiv":
        if args.device == "wisteria":
            dataset = load_dataset("EleutherAI/proof-pile-2", "arxiv")
        else:
            dataset = load_dataset("EleutherAI/proof-pile-2", "arxiv", cache_dir=f"{prefix}", trust_remote_code=True)
        member_dataset = dataset["train"]
        valid_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        non_member_dataset = concatenate_datasets([valid_dataset, test_dataset])
        if os.path.exists(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl"):
            # member_dataset = load_from_disk(
            #     f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
            # keep_in_memory=True)
            member_dataset = pickle.load(
                open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl",
                     "rb"))
            non_member_dataset = pickle.load(
                open(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_valid.pkl",
                     "rb"))
        else:
            # random_indices = random.sample(range(len(member_dataset)),
            #                                k=sample_num if sample_num < len(member_dataset) else len(
            #                                    member_dataset))
            # member_dataset = member_dataset.select(random_indices)
            # os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
            #             exist_ok=True)
            # member_dataset.save_to_disk(
            #     f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
            # member_dataset = load_from_disk(
            #     f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}")
            random_indices = random.sample(range(len(member_dataset)),
                                           k=sample_num if sample_num < len(member_dataset) else len(
                                               member_dataset))
            member_dataset = list(member_dataset.select(random_indices))
            non_member_dataset = list(non_member_dataset)
            os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}",
                        exist_ok=True)
            pickle.dump(member_dataset, open(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_train.pkl", "wb"))
            pickle.dump(non_member_dataset, open(
                f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{args.domain}/raw_data/{seed}_valid.pkl", "wb"))
        # merge valid and test
        #pdb.set_trace()
        # del dataset
        # validation_sampled = valid_dataset
        # test_sampled = test_dataset
        # non_member_dataset = concatenate_datasets([validation_sampled, test_sampled])
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
        os.makedirs(f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{min_length}_{max_length}_{args.select_method}/{args.domain}",
                    exist_ok=True)
        dataset.save_to_disk(
            f"{prefix}/dolma_absolute_filtered_dataset_{idx + 1}/{min_length}_{max_length}_{args.select_method}/{args.domain}")

