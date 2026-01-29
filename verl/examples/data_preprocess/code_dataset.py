# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the UltraFeedback best responses dataset to parquet format
"""

import argparse
import os
import json

import datasets
from datasets import Dataset

# from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_file", 
                       default="")

    args = parser.parse_args()

    data_source = "nvidia-OpenCodeInstruct"
    print(f"Loading the {args.input_file} dataset...", flush=True)
    
    # 读取 jsonl 文件
    data_list = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    # 转换为 Dataset
    dataset = Dataset.from_list(data_list)

    # 处理数据
    def process_fn(example, idx):
        instruction = example.get('input', '')
        best_response = example.get('output', '')
        
        data = {
            "source": "train0",
            "data_source": data_source,
            "messages": [
                {
                    "role": "user",
                    "content": instruction
                },
                {
                    "role": "assistant",
                    "content": best_response
                }
            ],
            "prompt": [
                {
                    "role": "user",
                    "content": instruction
                }
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": best_response},
            "extra_info": {
                "index": idx,
                "question": instruction,
                "answer": best_response,
                "split": "train"
            },
        }

        return data

    dataset = dataset.map(function=process_fn, with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 确保本地目录存在
    os.makedirs(local_dir, exist_ok=True)

    # 保存训练集和测试集
    # 训练集包含所有数据,测试集是训练集的第一条数据
    dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    print(f"处理完成! 共处理 {len(dataset)} 条数据")
    print(f"训练集: {len(dataset)} 条数据")
    print(f"训练集输出文件: {os.path.join(local_dir, 'train.parquet')}")
