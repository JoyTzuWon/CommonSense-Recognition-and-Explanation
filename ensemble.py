import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Union
import os


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path) #auto-tokenizer can apply to full fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict(model, tokenizer, texts: List[str], device, return_probs=True):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    if return_probs:
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    else:
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

def hard_voting(predictions: List[np.ndarray]) -> np.ndarray:
    predictions = np.array(predictions)  # shape: (n_models, n_samples)
    # 转置为 shape: (n_samples, n_models)，再按行投票
    majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    return majority_vote


def soft_voting(probabilities: List[np.ndarray]) -> np.ndarray:
    avg_prob = np.mean(probabilities, axis=0)  # shape: (n_samples, n_classes)
    final_preds = np.argmax(avg_prob, axis=1)
    return final_preds


def ensemble_predict(
    model_paths: List[str],
    texts: List[str],
    voting: str = "soft",
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> np.ndarray:
    """
    参数：
        model_paths: 模型路径列表
        texts: 待预测文本列表 [path1,path2,path3]
        voting: "soft" 或 "hard"
        device: use gpu
    返回：
        最终预测结果（聚类标签）：np.ndarray，形状 (n_samples,)
    """
    all_probs = []
    all_preds = []

    for path in model_paths:
        tokenizer, model = load_model(path)
        probs = predict(model, tokenizer, texts, device = device, return_probs=True)
        all_probs.append(probs)
        all_preds.append(np.argmax(probs, axis=1))

    if voting == "soft":
        return soft_voting(all_probs)
    elif voting == "hard":
        return hard_voting(all_preds)
    else:
        raise ValueError("voting must be either 'soft' or 'hard'")

if __name__ == "__main__":
    
    import utils
    import torch
    from ensemble import ensemble_predict
    from transformers import RobertaTokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained('roberta large') # 任意一个模型的 tokenizer 即可
    prompt_templates = [
        "{Sent1} </s> {Sent2}",
        "{Sent1} is more reasonable than {Sent2}",
        "Which is in common sense {Sent1} or {Sent2}?",
    ]
    template = prompt_templates[2]
    test_dataset = utils.preprocess_task1("./ALL data/test.csv", tokenizer, template, sep_token="</s>")
    texts = test_dataset["text"]
    labels = test_dataset["label"]
    model_dirs = ["./save_model/task1_roberta-large_template1", "./save_model/task1_roberta-large_template2", "./save_model/task1_roberta-large_template3"]
    from sklearn.metrics import accuracy_score
    
    soft_preds = ensemble_predict(model_dirs, texts, voting="soft", device = device)
    hard_preds = ensemble_predict(model_dirs, texts, voting="hard", device = device)
    print("Soft voting accuracy:", accuracy_score(labels, soft_preds))
    print("Hard voting accuracy:", accuracy_score(labels, hard_preds))