import os
import json
import matplotlib.pyplot as plt

train_losses = {}
grad_norms = {}
eval_losses = {}

fig, axs = plt.subplots(3, 1, figsize=(20, 30))

for dir in os.listdir(os.getcwd()):
    if dir.startswith("FFT") and os.path.isdir(dir):
        batch_size = int(dir.split("_")[-2])
        grad_acc = int(dir.split("_")[-1])

        with open(os.path.join(dir, "report.json"), "r") as f:
            report = json.load(f)

        train_records = [r for r in report if 'loss' in r]
        eval_records = [r for r in report if 'eval_loss' in r]
        train_loss = [r['loss'] for r in train_records]
        eval_loss = [r['eval_loss'] for r in eval_records]
        grad_norm = [r['grad_norm'] for r in train_records]

        train_losses[f"bs={batch_size}ga={grad_acc}"] = train_loss
        grad_norms[f"bs={batch_size}ga={grad_acc}"] = grad_norm
        eval_losses[f"bs={batch_size}ga={grad_acc}"] = eval_loss

        axs[0].plot(train_loss, label=f"bs={batch_size} ga={grad_acc}")
        axs[1].plot(grad_norm, label=f"bs={batch_size} ga={grad_acc}")
        axs[2].plot(eval_loss, label=f"bs={batch_size} ga={grad_acc}")

axs[0].set_title("Training Loss")
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].set_title("Gradient Norm")
axs[1].set_xlabel("Steps")
axs[1].set_ylabel("Gradient Norm")
axs[1].legend()

axs[2].set_title("Evaluation Loss")
axs[2].set_xlabel("Steps")
axs[2].set_ylabel("Loss")
axs[2].legend()



plt.savefig('report.png')

