import matplotlib.pyplot as plt
import os

def plot_training_curves(t_losses, t_accs, v_losses, v_accs, save_path="../outputs/plots/model_training.png"):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    ax_left.plot(t_losses, label="Train Loss")
    ax_left.plot(v_losses, label="Val Loss")
    ax_left.set_title("Loss")
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Loss")
    ax_left.legend()

    ax_right.plot(t_accs, label="Train Acc")
    ax_right.plot(v_accs, label="Val Acc")
    ax_right.set_title("Accuracy")
    ax_right.set_xlabel("Epoch")
    ax_right.set_ylabel("%")
    ax_right.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()