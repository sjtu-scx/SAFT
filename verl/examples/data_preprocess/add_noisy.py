import pandas as pd
import numpy as np
import os
import json

# ==========================================
# 配置部分
# ==========================================
INPUT_FILE = "../data/numina_cot/train.parquet"
OUTPUT_DIR = "noisy_datasets"
NOISE_RATIOS = [0.1, 0.2, 0.3, 0.5]
SEED = 42

# 关键：指定 SFT 训练真正使用的 Target 列名
# 在你的结构中，通常 SFT 的标签来源是顶层的 'solution' 或 'response' 列
# 或者是 reward_model['ground_truth']
# 脚本会同时修改：
# 1. 顶层 Target 列 (用于一般 SFT)
# 2. extra_info['answer'] (用于你提到的特殊 SFT 逻辑)
# 3. reward_model['ground_truth'] (用于保持一致性)
TARGET_COL_NAME = "solution" # 请确认 parquet 中存储完整回答的列名，Numina通常叫 'solution'

# ==========================================
# 核心逻辑
# ==========================================

def inject_noise_nested(df, ratio, seed):
    """
    对 DataFrame 进行原位修改，注入噪声。
    支持修改顶层列以及嵌套在 extra_info 和 reward_model 中的字段。
    """
    n_samples = len(df)
    n_noise = int(n_samples * ratio)
    
    if n_noise == 0:
        return df
    
    np.random.seed(seed)
    
    # 1. 选择要被污染的索引
    noise_indices = np.random.choice(df.index, size=n_noise, replace=False)
    
    # 2. 获取这些位置的真实数据
    # 我们不仅要打乱 Solution，最好连同 extra_info 里的 answer 一起打乱，保持错得"一致"
    
    # 获取顶层 solution (如果存在)
    if TARGET_COL_NAME in df.columns:
        original_solutions = df.loc[noise_indices, TARGET_COL_NAME].values
        # 打乱
        shuffled_solutions = np.random.permutation(original_solutions)
        # 赋值回去
        df.loc[noise_indices, TARGET_COL_NAME] = shuffled_solutions
    
    # 3. 处理嵌套字段 (extra_info, reward_model)
    # pandas 处理嵌套字典比较慢，我们需要逐行处理或使用 apply
    # 为了效率，我们先提取出来，打乱，再填回去
    
    # 提取 extra_info 列 (假设是 dict 或 struct)
    # 注意：如果读取后是 None 或其他类型，需要做空值检查
    # 这里假设 extra_info 是完整的字典
    
    # 策略：我们直接交换整行数据的 target 部分，把 A 的答案给 B
    # 为了实现这一点，我们可以创建一个映射： idx -> shuffled_idx
    shuffled_indices = np.random.permutation(noise_indices)
    
    # 这是一个高效的方法：直接把 shuffled_indices 对应行的数据 赋值给 noise_indices
    # 但我们只交换 Target 相关字段，保留 Question 不变！
    
    print(f"  正在处理嵌套字段 (Ratio={ratio})...")
    
    # 必须显式复制，否则会 SettingWithCopyWarning
    # 我们遍历需要修改的每一行
    for original_idx, source_idx in zip(noise_indices, shuffled_indices):
        
        # === A. 获取“来源行” (Source) 的答案信息 ===
        source_row = df.loc[source_idx]
        
        # 1. 获取 Source 的 extra_info['answer']
        source_extra = source_row['extra_info']
        if isinstance(source_extra, dict):
            fake_answer = source_extra.get('answer', None)
        else:
            # 如果是 numpy array 或其他对象
            fake_answer = None # 视具体情况处理

        # 2. 获取 Source 的 reward_model['ground_truth']
        source_reward = source_row['reward_model']
        if isinstance(source_reward, dict):
            fake_ground_truth = source_reward.get('ground_truth', None)
        else:
            fake_ground_truth = None
            
        # === B. 修改“目标行” (Original) ===
        # 注意：在 pandas 中修改嵌套对象需要非常小心，建议先取出对象，修改后整体赋值
        
        target_extra = df.at[original_idx, 'extra_info']
        target_reward = df.at[original_idx, 'reward_model']
        
        # 深拷贝以防引用问题 (如果 dict 是共用的，虽然一般不会)
        if isinstance(target_extra, dict):
            target_extra = target_extra.copy()
            target_extra['answer'] = fake_answer # 注入错误的 answer
            # 重要：标记该样本为 Noisy，方便后续分析！
            target_extra['is_noisy_sample'] = True 
            df.at[original_idx, 'extra_info'] = target_extra
            
        if isinstance(target_reward, dict):
            target_reward = target_reward.copy()
            target_reward['ground_truth'] = fake_ground_truth # 注入错误的 ground_truth
            df.at[original_idx, 'reward_model'] = target_reward

    return df

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"读取原始数据: {INPUT_FILE}")
    df_base = pd.read_parquet(INPUT_FILE)
    print(f"总行数: {len(df_base)}")

    # 检查列结构
    print("Columns:", df_base.columns.tolist())
    
    for ratio in NOISE_RATIOS:
        print(f"\n=== 生成噪声比例 {ratio} 的数据集 ===")
        df_noisy = df_base.copy()
        
        # 注入噪声
        df_noisy = inject_noise_nested(df_noisy, ratio, SEED)
        
        # 保存
        filename = f"train_noise_{ratio}.parquet"
        save_path = os.path.join(OUTPUT_DIR, filename)
        df_noisy.to_parquet(save_path)
        print(f"已保存: {save_path}")
        
        # 验证一下
        print("  验证数据一致性 (随机抽取一个噪声样本):")
        # 找到被修改的样本
        # 这种嵌套检查比较慢，只做一次演示
        sample_noisy = df_noisy[df_noisy['extra_info'].apply(lambda x: x.get('is_noisy_sample', False))].head(1)
        if not sample_noisy.empty:
            idx = sample_noisy.index[0]
            print(f"  Sample ID: {idx}")
            print(f"  Original Question: {sample_noisy.iloc[0]['extra_info']['question'][:50]}...")
            print(f"  Corrupted Answer : {sample_noisy.iloc[0]['extra_info']['answer'][:50]}...")
            print(f"  (这个 Answer 应该是来自其他问题的)")

if __name__ == "__main__":
    main()