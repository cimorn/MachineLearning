import torch
import os
import pandas as pd
import json

def save_agent(agent, path):
    """模型权重保存与 JSON 日志持久化"""
    
    # 保存模型
    torch.save(agent.q_net.state_dict(), os.path.join(path, "Q_net.pth"))
    if hasattr(agent, 'policy_net'):
        torch.save(agent.policy_net.state_dict(), os.path.join(path, "policy_net.pth"))
    print(f"agent------>{path}\n")
    

def save_results(results, path, filename):
    # 拼接最终的保存路径
    json_path = os.path.join(path, f"{filename}.json")
    csv_path = os.path.join(path, f"{filename}.csv")

    # 保存JSON文件（无异常捕获，出错直接抛出原生异常）
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 校验results格式并处理CSV数据
    if isinstance(results, dict):
        csv_results = [{k: v for k, v in results.items() if k != 'all_eval_rewards'}]
    elif isinstance(results, list):
        csv_results = [{k: v for k, v in item.items() if k != 'all_eval_rewards'} for item in results]
    else:
        raise ValueError("results必须是字典或列表格式！")
    
    # 保存CSV文件（无异常捕获，出错直接抛出原生异常）
    pd.DataFrame(csv_results).to_csv(csv_path, index=False)
    print(f"{filename}.json------>{json_path}")
    print(f"{filename}.csv------>{csv_path}\n")

