#from transformers.utils import logging
#logging.set_verbosity_error()  # 只显示错误信息，屏蔽所有 warning/info/debug
#import os
#os.environ["DISABLE_PROGRESS_BAR"] = "1"
#from datasets.utils.logging import disable_progress_bar
#disable_progress_bar()
import utils
import numpy as np
import torch
from ensemble import ensemble_predict
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
torch.cuda.empty_cache()
def evaluate_single_model(model_path, texts, labels, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    acc = accuracy_score(labels, preds)
    return acc

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert') # 任意一个模型的 tokenizer 即可
    #prompt_templates = [
    #    "{Sent1} </s> {Sent2}",
    #    "{Sent1} is more reasonable than {Sent2}",
    #    "Which is in common sense {Sent1} or {Sent2}?",
    #]
    prompt_templates = [
        "{Incorrect} </s> {R1} </s> {R2} </s> {R3}",
        "{Incorrect} is against common sense because {R1} or {R2} or {R3}.",
        "If [Correct] is in common sense then {Incorrect} is against common sense because {R1} or {R2} or {R3}"
    ]
    template = prompt_templates[1]
    
    model_dirs = ["./save_model/task2_roberta-large_template1", "./save_model/task2_xlnet-large-cased_template1", "./save_model/task2_bert-large-uncased_template1"]
    N_RUNS = 5
    print("==== Task 2 Template 1 ====")
    # --- Soft & Hard Voting 多次运行 ---
    for voting in ["soft", "hard"]:
        print(f"\n==== {voting.upper()} Voting ({N_RUNS} runs) ====")
        accs = []
        for i in range(N_RUNS):
            test_dataset = utils.preprocess_task2("./ALL data/test.csv", tokenizer, template, sep_token="[SEP]")
            texts = test_dataset["text"]
            labels = test_dataset["label"]
            preds = ensemble_predict(model_dirs, texts, voting=voting, device=device)
            acc = accuracy_score(labels, preds)
            accs.append(acc)
            #print(f"Run {i+1}: {acc:.4f}")
        print(f">>> Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    # --- 单模型准确率多次运行 ---
    print(f"\n==== Individual Model Accuracy ({N_RUNS} runs each) ====")
    for path in model_dirs:
        model_name = os.path.basename(path)
        accs = []
        for i in range(N_RUNS):
            test_dataset = utils.preprocess_task2("./ALL data/test.csv", tokenizer, template, sep_token="[SEP]")
            texts = test_dataset["text"]
            labels = test_dataset["label"]
            acc = evaluate_single_model(path, texts, labels, device)
            accs.append(acc)
        print(f"{model_name}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    


