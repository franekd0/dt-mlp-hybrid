import matplotlib.pyplot as plt
import numpy as np


def plot_loss_summary(loss_histories, experiment_name, save_path=None):
    if not loss_histories:
        print("Brak historii loss do narysowania.")
        return

    min_len = min(len(h) for h in loss_histories)

    loss_data = np.array([h[:min_len] for h in loss_histories])

    mean_loss = np.mean(loss_data, axis=0)
    std_loss = np.std(loss_data, axis=0)
    epochs = range(1, len(mean_loss) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, mean_loss, label='Mean Loss', color='#2980b9', linewidth=2)

    plt.fill_between(epochs,
                     mean_loss - std_loss,
                     mean_loss + std_loss,
                     color='#3498db',
                     alpha=0.25,
                     label='± Std Dev')

    plt.title(f'Training Loss Stability | {experiment_name}\n(Avg over {len(loss_histories)} runs)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (CrossEntropy)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Zapisano wykres loss: {save_path}")
    else:
        plt.show()

    plt.close()