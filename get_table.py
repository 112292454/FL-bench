import os
import pandas as pd
import re

def extract_accuracy_from_log(log_path):
    """从日志文件中提取 before 和 after 的准确率"""
    with open(log_path, 'r') as file:
        content = file.read()
        before_match = re.search(r"before fine-tuning: ([\d.]+)%", content)
        after_match = re.search(r"after fine-tuning: ([\d.]+)%", content)

        if before_match and after_match:
            before_accuracy = float(before_match.group(1))
            after_accuracy = float(after_match.group(1))
            return before_accuracy, after_accuracy
    return None, None

def create_accuracy_tables(base_path):
    """创建包含算法名称和数据集名称的两张准确率表格: before 和 after"""
    before_data = {}
    after_data = {}

    # 遍历路径的第一层（算法名称）
    for method_name in os.listdir(base_path):
        method_path = os.path.join(base_path, method_name)
        if not os.path.isdir(method_path):
            continue

        # 遍历路径的第二层（数据集名称）
        for dataset_name in os.listdir(method_path):
            dataset_path = os.path.join(method_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue

            # 遍历路径的第三层（时间戳）
            for timestamp in os.listdir(dataset_path):
                timestamp_path = os.path.join(dataset_path, timestamp)
                log_file = os.path.join(timestamp_path, "main.log")

                if os.path.isfile(log_file):
                    before, after = extract_accuracy_from_log(log_file)

                    if before is not None and after is not None:
                        if method_name not in before_data:
                            before_data[method_name] = {}
                        if method_name not in after_data:
                            after_data[method_name] = {}

                        before_data[method_name][dataset_name] = before
                        after_data[method_name][dataset_name] = after

    # 创建 DataFrames
    before_df = pd.DataFrame(before_data)
    after_df = pd.DataFrame(after_data)

    # 转置
    before_df = before_df.T
    after_df = after_df.T

    #按照以下method顺序排列：
    # methods=['FedMD','pfedsim','perFedAvg','pFedHN','pFedMe','FedEM','Floco',]  # 4090*4
    # methods=['FedAP','kNNPer','FedProto','FedPAC','MetaFed','FedALA','FedAS','pFedFDA','FedFomo','FedBN','CFL',]  # 3090*8
    # methods=['pFedLA','FedBabu','PeFLL','Ditto',]   # a800*3
    method_order = [
    # 4090*4
    'FedMD', 'pfedsim', 'perFedAvg', 'pFedHN', 'pFedMe', 'FedEM', 'Floco',
    # 3090*8
    'FedAP', 'kNNPer', 'FedProto', 'FedPAC', 'MetaFed', 'FedALA', 'FedAS', 'pFedFDA', 'FedFomo', 'FedBN', 'CFL',
    # a800*3
    'pFedLA', 'FedBabu', 'PeFLL', 'Ditto']

    method_order = [m.lower() for m in method_order]
    before_df = before_df.reindex(method_order)
    after_df = after_df.reindex(method_order)

    return before_df, after_df

# 示例路径
base_path = "./out"          # 结果路径
before_table, after_table = create_accuracy_tables(base_path)

# 显示表格
print("Before Accuracy Table:")
print(before_table)
print("\n After Accuracy Table:")
print(after_table)

# 保存表格到文件
before_table.to_csv("before_accuracy_table.csv")
after_table.to_csv("after_accuracy_table.csv")