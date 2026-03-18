import os
import sys
sys.path.append(os.getcwd()) # 回到根目录
from run.config import Path

def create_output_dirs():
    # 遍历第一层配置
    for main_key in Path.keys():
        # 处理嵌套字典（data/model/results）
        if isinstance(Path[main_key], dict):
            for dir_path in Path[main_key].values():
                os.makedirs(dir_path, exist_ok=True)
        # 处理单层路径（logs）
        else:
            os.makedirs(Path[main_key], exist_ok=True)
    print("===== 所有目录创建完成 =====")

if __name__ == "__main__":
    create_output_dirs()