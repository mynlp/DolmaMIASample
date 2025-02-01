#!/bin/bash
#SBATCH --job-name=datacollection
#SBATCH --partition=010-partition
#SBATCH --gres=gpu:1              # 请求8个GPU
#SBATCH --nodes=1                  # 确保在一个节点上运行
#SBATCH --ntasks=8                 # 并行启动8个任务
#SBATCH --output=datacollection.log
#SBATCH --error=datacollection.err
#SBATCH --cpus-per-task=8          # 每个任务使用的CPU数量

export TRANSFORMERS_CACHE=/home/user10003/.cache
export HF_HOME=/home/user10003/.cache

# 定义并行化函数
source ~/anaconda3/bin/activate base
echo "Current working directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Environment Variables:"
printenv

python create_mia_dataset.py  --device beyondai --domain arxiv --batch_size 100