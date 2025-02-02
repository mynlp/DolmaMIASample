#!/usr/bin/env python3
import os
import shutil

def reorganize_dataset(src_dir, dest_dir, skip_intervals=None):
    """
    将原始数据按照 domain 分组重组成：
      目标目录 dest_dir 下首先按 domain 分类，
      每个 domain 目录内包含各个区间（例如 5_100, 100_200, 200_300, …, 900_1000）
    参数：
      src_dir       源数据目录，如 "dolma_absolute_filtered_dataset_1"
      dest_dir      重组后目标目录，如 "dolma"
      skip_intervals 需要跳过的区间列表，这里置为空列表表示不跳过任何区间，包含 "200_300"
    """
    if skip_intervals is None:
        skip_intervals = []  # 为空列表表示不跳过任何区间

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"创建目标目录：{dest_dir}")

    # 遍历所有以 "_truncated" 结尾的目录
    for truncated_folder in os.listdir(src_dir):
        truncated_path = os.path.join(src_dir, truncated_folder)
        if not os.path.isdir(truncated_path):
            continue  # 仅处理文件夹

        if not truncated_folder.endswith("_truncated"):
            print(f"跳过非 _truncated 文件夹：{truncated_folder}")
            continue

        # 通过去除后缀得到区间名称
        interval_name = truncated_folder[:-len("_truncated")]
        if interval_name in skip_intervals:
            print(f"跳过区间：{interval_name}")
            continue

        print(f"处理区间：{interval_name}")

        # 遍历该区间下的所有 domain 文件夹
        for domain in os.listdir(truncated_path):
            domain_src = os.path.join(truncated_path, domain)
            if not os.path.isdir(domain_src):
                continue

            # 在目标目录下以 domain 名称创建文件夹（若不存在则创建）
            domain_dest = os.path.join(dest_dir, domain)
            if not os.path.exists(domain_dest):
                os.makedirs(domain_dest)
                print(f"创建 domain 文件夹：{domain_dest}")

            # 在该 domain 文件夹内创建区间目录
            interval_dest = os.path.join(domain_dest, interval_name)
            print(f"复制 {domain_src} 到 {interval_dest}")
            shutil.copytree(domain_src, interval_dest, dirs_exist_ok=True)
            # 如果你更希望使用“移动”而非复制，可使用：
            # shutil.move(domain_src, interval_dest)

if __name__ == "__main__":
    # 设定源数据文件夹，本示例要求该文件夹与脚本处于同一目录，必要时请调整路径
    source_directory = "dolma_absolute_filtered_dataset_2"
    # 指定重组后数据存放的目标文件夹
    destination_directory = "dolma_dataset_truncated_2"
    # 若你需要跳过某某区间，在此处列出即可；这里设为空列表，表示不跳过任何区间，包括 200_300
    skip_intervals = []

    reorganize_dataset(source_directory, destination_directory, skip_intervals)
    print("数据重组完成！")