# utils.py
import os
import csv
import pandas as pd
import random
import numpy as np
from datasets import Dataset
from transformers import Trainer
from datetime import datetime
import matplotlib.pyplot as plt


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}


def preprocess_task1(file, tokenizer, prompt_template=None, sep_token="<sep>"):
    # 文件存在性检查
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Data file not found: {file}")

    # 尝试读取
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise RuntimeError(f"Failed to read file '{file}': {e}")

    data = []
    if prompt_template is None:
        prompt_template = "{Sent1} <sep> {Sent2}"

    for _, row in df.iterrows():
        correct = row["Correct Statement"]
        incorrect = row["Incorrect Statement"]
        p = random.random()

        if p < 0.5:
            first, second = correct, incorrect
            label = 0
        else:
            first, second = incorrect, correct
            label = 1

        text = prompt_template.format(Sent1=first, Sent2=second)
        text = text.replace("<sep>", sep_token)
        data.append((text, label))

    random.shuffle(data)
    texts, labels = zip(*data)
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    return tokenized_dataset


def preprocess_task2(file, prompt_template=None, sep_token="<sep>"):
    # 文件存在性检查
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Data file not found: {file}")

    # 尝试读取
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise RuntimeError(f"Failed to read file '{file}': {e}")

    texts = []
    labels = []

    if prompt_template is None:
        # 默认使用 sep 拼接
        prompt_template = "{Incorrect} <sep> {R1} <sep> {R2} <sep> {R3}"

    for _, row in df.iterrows():
        correct = row["Correct Statement"]
        incorrect = row["Incorrect Statement"]
        reasons = [
            row["Right Reason1"],         # 正确理由（位置0）
            row["Confusing Reason1"],     # 错误理由1
            row["Confusing Reason2"],     # 错误理由2
        ]

        # 打乱理由顺序
        idx = [0, 1, 2]
        random.shuffle(idx)
        correct_pos = idx.index(0)

        # 替换模板变量
        if "Correct" in prompt_template:
            formatted_prompt = prompt_template.format(
                Correct=correct,
                Incorrect=incorrect,
                R1=reasons[idx[0]],
                R2=reasons[idx[1]],
                R3=reasons[idx[2]]
            )
        else:
            formatted_prompt = prompt_template.format(
                Incorrect=incorrect,
                R1=reasons[idx[0]],
                R2=reasons[idx[1]],
                R3=reasons[idx[2]]
            )

        # 替换 sep token
        formatted_prompt = formatted_prompt.replace("<sep>", sep_token)

        texts.append(formatted_prompt)
        labels.append(correct_pos)

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def get_log_path(log_dir="./logs"):
    """
    生成训练日志和验证日志两个 CSV 路径，自动带时间戳
    返回 (train_log_path, eval_log_path)
    """
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_log = os.path.join(log_dir, f"log_{now}_train.csv")
    eval_log = os.path.join(log_dir, f"log_{now}_eval.csv")
    return train_log, eval_log

def write_log_to_csv(log_dict, filepath):
    """将日志字典写入 CSV 文件（如不存在则自动添加表头）"""
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)

class LoggingTrainer(Trainer):
    def __init__(self, *args, train_log_path=None, eval_log_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_log_path = train_log_path
        self.eval_log_path = eval_log_path

    def log(self, logs):
        super().log(logs)
        if "loss" in logs and self.train_log_path:
            write_log_to_csv(logs, self.train_log_path)
        elif "eval_loss" in logs and self.eval_log_path:
            write_log_to_csv(logs, self.eval_log_path)
