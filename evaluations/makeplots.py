import re
import matplotlib.pyplot as plt

log_path = "/net/noble/vol1/home/ddesai22/casanovo_train_val.log"

# Load the log lines
with open(log_path, 'r') as f:
    lines = f.readlines()

# Initialize lists
steps = []         # For training steps
train_losses = []
val_steps = []     # For validation steps
val_losses = []

# regex pattern to extract step, train_loss, val_loss from log
pattern = re.compile(r": (\d+)\s+(nan|\d+\.\d+)\s+(nan|\d+\.\d+)")

for line in lines:
    match = pattern.search(line)
    if match:
        step = int(match.group(1))
        train_loss = match.group(2)
        val_loss = match.group(3)
        
        if train_loss != "nan":
            steps.append(step)
            train_losses.append(float(train_loss))

        if val_loss != "nan":
            val_steps.append(step)
            val_losses.append(float(val_loss))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Training Loss Plot
plt.figure()
plt.plot(steps, train_losses, label='Train Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss per Step')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
plt.savefig("training_loss_plot.png")
plt.show()

# Validation Loss Plot
plt.figure()
plt.plot(val_steps, val_losses, label='Validation Loss', color='orange')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Validation Loss per Step')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
plt.savefig("validation_loss_plot.png")
plt.show()
