import os
from utils import plot_train_loss, plot_eval_accuracy

train_logs = ["./logs/task1_template0_train.csv", "./logs/task1_template1_train3.csv", "./logs/task1_template2_train2.csv"]
eval_logs = ["./logs/task1_template0_eval.csv", "./logs/task1_template1_eval3.csv", "./logs/task1_template2_eval2.csv"]
labels = ["template0", "template1", "template2"]
train_save_path = "./plots/task2_train"
test_save_path = "./plots/task2_eval"

# train_logs = ["./logs/task2_template0_train.csv", "./logs/task2_template1_train.csv", "./logs/task2_template2_train.csv"]
# eval_logs = ["./logs/task2_template0_eval.csv", "./logs/task2_template1_eval.csv", "./logs/task2_template2_eval.csv"]
# labels = ["template0", "template1", "template2"]
# train_save_path = "./plots/task2_train"
# test_save_path = "./plots/task2_eval"

# train_logs = ["./logs/task1_template1_train.csv", "./logs/task1_template1_train2.csv", "./logs/task1_template1_train3.csv"]
# eval_logs = ["./logs/task1_template1_eval.csv", "./logs/task1_template1_eval2.csv", "./logs/task1_template1_eval3.csv"]
# labels = ["1", "2", "3"]
# train_save_path = "./plots/task1_t1_train"
# test_save_path = "./plots/task1_t1_eval"

os.makedirs("./plots", exist_ok=True)
plot_train_loss(train_logs, labels, title="Task1 Train Loss Comparison", save_path=train_save_path)
plot_eval_accuracy(eval_logs, labels, title="Task1 Validation Accuracy Comparison", save_path=test_save_path)
