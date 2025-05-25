# load_test.py
import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import utils
from train_setting import training_args

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    # load model
    model_path = "./save_model/task1_xlnet-large-cased_template1_3"
    model = XLNetForSequenceClassification.from_pretrained(model_path)
    tokenizer = XLNetTokenizer.from_pretrained(model_path)
    print("Using model", model_path)

    prompt_templates = [
        "{Sent1} <sep> {Sent2}",
        "{Sent1} is more reasonable than {Sent2}",
        "Which is in common sense {Sent1} or {Sent2}",
    ]
    template = prompt_templates[1]
    print("Using prompt template:", template)

    train_dataset = []
    valid_dataset = []
    test_dataset = utils.preprocess_task1("./ALL data/test.csv", tokenizer, template)

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

    test_results = trainer.predict(test_dataset)
    print(test_results.metrics)
    print(f"Test Accuracy: {test_results.metrics['test_accuracy']:.4f}")



