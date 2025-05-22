# task1_xlnet_model_train.py
import os
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import utils
from train_setting import training_args

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    # 自动创建训练输出目录、日志目录
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)

    # 可替换为其它模型，如 Roberta、BERT
    model_path = "./xlnet_model"
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = XLNetForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
        dropout=0.1,
    ).to(device)
    tokenizer = XLNetTokenizer.from_pretrained(model_path)

    prompt_templates = [
        "{Sent1} <sep> {Sent2}",
        "{Sent1} is more reasonable than {Sent2}",
        "Which is in common sense {Sent1} or {Sent2}?",
    ]
    template = prompt_templates[1]
    print("Using prompt template:", template)

    train_dataset = utils.preprocess_task1("./ALL data/train.csv", tokenizer, template)
    print("Train samples:", len(train_dataset))
    valid_dataset = utils.preprocess_task1("./ALL data/dev.csv", tokenizer, template)
    print("Validation samples:", len(valid_dataset))
    test_dataset = utils.preprocess_task1("./ALL data/test.csv", tokenizer, template)
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
    print("Training finished.")
    # print(trainer.state.log_history)

    # save model
    model.save_pretrained("./xlnet_model")
    tokenizer.save_pretrained("./xlnet_model")
    # load model
    # model = XLNetForSequenceClassification.from_pretrained("./xlnet_model")
    # tokenizer = XLNetTokenizer.from_pretrained("./xlnet_model")

    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)
    print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")
