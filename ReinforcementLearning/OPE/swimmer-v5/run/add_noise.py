import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from tools.seed import set_seed
from run.config import Path, global_seed

def add_noise_to_data(csv_name: str = "train_data.csv", output_name: str = "val_data.csv", noise_std: float = 0.1) -> None:
    set_seed(global_seed)
    
    df = pd.read_csv(os.path.join(Path.data.raw, csv_name))
    
    df_noisy = df.copy()
    
    noise_columns = ["reward", "ns1", "ns2", "ns3", "ns4", "ns5", "ns6", "ns7", "ns8"]
    
    for col in noise_columns:
        if col in df_noisy.columns:
            noise = np.random.normal(0, noise_std, size=len(df_noisy))
            df_noisy[col] = df_noisy[col] + noise
    
    df_noisy.to_csv(os.path.join(Path.data.raw, output_name), index=False)
    print(f"已为 {csv_name} 添加噪声并保存为 {output_name}，标准差: {noise_std}")
