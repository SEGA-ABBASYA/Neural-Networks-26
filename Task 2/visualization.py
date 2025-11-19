import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def _compute_confusion_matrix(model, X_test, y_test, y_pred=None, num_classes=3):
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if y_test.ndim > 1:
        y_test_idx = np.argmax(y_test, axis=1)
    else:
        y_test_idx = y_test.astype(int).reshape(-1)

    # Obtain predictions (either provided or from model) and normalize to indices
    if y_pred is None:
        y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)

    if y_pred.ndim > 1:
        if np.issubdtype(y_pred.dtype, np.integer) and y_pred.shape[1] == 1:
            y_pred_idx = y_pred.reshape(-1)
        else:
            y_pred_idx = np.argmax(y_pred, axis=1)
    else:
        y_pred_idx = y_pred.astype(int).reshape(-1)

    # build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_test_idx, y_pred_idx):
        cm[int(t)][int(p)] += 1

    accuracy = np.mean(y_test_idx == y_pred_idx) * 100.0
    return cm, accuracy, y_test_idx, y_pred_idx


def visualize_data(X_train, y_train, X_test, y_test, y_pred, model, class1="Adelie", class2="Chinstrap", class3="Gentoo"):
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    cm, accuracy_value, y_test_idx, y_pred_idx = _compute_confusion_matrix(model, X_test, y_test, y_pred, num_classes=3)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.8, 1.5, 1.2], height_ratios=[1, 1])

    # confusion matrix
    ax2 = fig.add_subplot(gs[0, 0])
    im = ax2.imshow(cm, cmap='Blues', alpha=0.9)
    ax2.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels([class1, class2, class3])
    ax2.set_yticklabels([class1, class2, class3])
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")

    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=20, fontweight="bold")

    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # parameters table
    ax3 = fig.add_subplot(gs[0, 1:])
    ax3.axis('off')

    total_params = 0
    for w in getattr(model, "weights", []):
        total_params += int(np.prod(np.asarray(w).shape))
    if getattr(model, "use_bias", False):
        for b in getattr(model, "biases", []):
            total_params += int(np.prod(np.asarray(b).shape))

    table_data = [
        ["Hidden Layers", str(getattr(model, "num_hidden_layers", "N/A"))],
        ["Neurons per Layer", str(getattr(model, "neurons_per_layer", "N/A"))],
        ["Activation", str(getattr(model, "activation_function", "N/A")).capitalize()],
        ["Learning Rate", f"{getattr(model, 'learning_rate', 0.0):.4f}"],
        ["Epochs", str(getattr(model, "n_epochs", "N/A"))],
        ["Bias Used", "Yes" if getattr(model, "use_bias", False) else "No"],
        ["Total Parameters", str(total_params)],
        ["Final MSE", f"{getattr(model, 'errors_history', [np.nan])[-1]:.6f}"],
        ["Test Accuracy", f"{accuracy_value:.2f}%"],
    ]

    table = ax3.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        cellLoc='left',
        loc='center',
        colWidths=[0.5, 0.5]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.8)

    cells = table.get_celld()
    num_cols = len(table_data[0])

    for col in range(num_cols):
        cell = cells.get((0, col))
        if cell:
            cell.set_facecolor('#0EA5E9')
            cell.set_text_props(weight='bold', color='white')

    # save
    plt.suptitle("Penguin Species Classification - Backpropagation Results",
                 fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout()

    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    filename = f"penguin_backprop_result_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(vis_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {filepath}")
    return filepath