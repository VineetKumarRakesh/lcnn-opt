import re
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files
from io import StringIO

uploaded = files.upload()

model_data = {}

# Parse logs
for filename, content in uploaded.items():
    text = content.decode()
    epochs, val_accs, lrs = [], [], []
    print(f"Processing {filename}") # Added for debugging
    for line in text.splitlines():
        epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
        valacc_match = re.search(r'Val Acc:\s*([\d.]+)%?', line)
        lr_match = re.search(r'\[DEBUG\] lr = ([\d.eE+-]+)', line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))
            print(f"Matched epoch: {epoch_match.group(1)}") # Added for debugging
        if valacc_match:
            val_accs.append(float(valacc_match.group(1)))
            print(f"Matched val_acc: {valacc_match.group(1)}") # Added for debugging
        if lr_match:
            lrs.append(float(lr_match.group(1)))
            print(f"Matched lr: {lr_match.group(1)}") # Added for debugging
    model_name = filename.replace('.txt', '')
    # Ensure all lists have the same length before creating DataFrame
    min_len = min(len(epochs), len(val_accs), len(lrs))
    if min_len > 0:
        model_data[model_name] = pd.DataFrame({
            'Epoch': epochs[:min_len],
            'Val Acc': val_accs[:min_len],
            'LR': lrs[:min_len]
        })
    else:
        print(f"No data parsed for {filename}") # Added for debugging
        model_data[model_name] = pd.DataFrame(columns=['Epoch', 'Val Acc', 'LR'])


# Plot accuracy vs learning rate
plt.figure()
for model, df in model_data.items():
    if not df.empty:
        plt.plot(df['LR'], df['Val Acc'], label=model)
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy (%)')
plt.title('Accuracy vs Learning Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_lr.png')
plt.show()

# Plot accuracy vs epochs
plt.figure()
for model, df in model_data.items():
    if not df.empty:
        plt.plot(df['Epoch'], df['Val Acc'], label=model)
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_epochs.png')
plt.show()

# Ask user if they want to download the plots
download_choice = input("Do you want to download the plots? (yes/no): ").lower()

if download_choice == 'yes':
    files.download('accuracy_vs_lr.png')
    files.download('accuracy_vs_epochs.png')
    print("Plots downloaded.")
else:
    print("Plots not downloaded.")