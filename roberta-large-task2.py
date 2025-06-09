import os
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForMultipleChoice, RobertaPreTrainedModel, RobertaModel,RobertaForSequenceClassification
import utils
from train_setting import training_args

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    # 自动创建训练输出目录、日志目录
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)

    # 可替换为其它模型，如 Roberta、BERT
    model_path = "roberta large"
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = RobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
    ).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    prompt_templates = [
        "{Incorrect} </s> {R1} </s> {R2} </s> {R3}",
        "{Incorrect} is against common sense because {R1} or {R2} or {R3}.",
        "If [Correct] is in common sense then {Incorrect} is against common sense because {R1} or {R2} or {R3}"
    ]

    template = prompt_templates[2]
    print("Using prompt template:", template)

    train_dataset = utils.preprocess_task2("./ALL data/train.csv", tokenizer, template, sep_token = "</s>")
    print("Train samples:", len(train_dataset))
    valid_dataset = utils.preprocess_task2("./ALL data/dev.csv", tokenizer, template, sep_token = "</s>")
    print("Validation samples:", len(valid_dataset))
    test_dataset = utils.preprocess_task2("./ALL data/test.csv", tokenizer, template, sep_token = "</s>")
    print("Test samples:", len(test_dataset))

    train_log_path, eval_log_path = utils.get_log_path()

    trainer = utils.LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=utils.compute_metrics,
        train_log_path=train_log_path,
        eval_log_path=eval_log_path
    )

    trainer.train()
    print("-"*20)
    print("Training finished.")
    print("-"*20)
    # print(trainer.state.log_history)

    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)
    print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")

    # save model
    save_model_path = "./save_model/task2_roberta-large_template3"
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print("save model to", save_model_path)