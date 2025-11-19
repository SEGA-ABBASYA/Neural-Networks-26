import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_data(X_train, y_train, X_test, y_test, y_pred, model, class1 = "Adelie", class2 = "Chinstrap", class3 = "Gentoo"):
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred).flatten()

    if y_test.ndim > 1: # Convert target to class indices
        y_test = np.argmax(y_test, axis = 1)

    accuracy_value = np.mean(y_test == y_pred) * 100
    cm = np.zeros((3, 3), dtype = int)

    # Confusion matrix (3×3)
    for true, pred in zip(y_test, y_pred):
        cm[true][pred] += 1
    fig = plt.figure(figsize = (18, 10))
    gs = GridSpec(2, 4, figure = fig, width_ratios = [2.5, 1.5, 1.5, 1.2], height_ratios = [1, 1])

    # 7esbet el Loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(model.errors_history) + 1)
    ax1.plot(epochs, model.errors_history, color = '#0EA5E9', linewidth = 3, marker = 'o', markersize = 4)
    ax1.set_title("Training Loss Over Time", fontsize = 16, fontweight = 'bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error")
    ax1.grid(True, alpha = 0.4)
    ax1.set_facecolor('#f8f9fa')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Rasmet el confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(cm, cmap = 'Blues', alpha = 0.9)
    ax2.set_title("Confusion Matrix", fontsize = 16, fontweight = 'bold')
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels([class1, class2, class3])
    ax2.set_yticklabels([class1, class2, class3])
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    for i in range(3): # 3ashan el arkam tetketeb gowa el matrix
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax2.text(j, i, str(cm[i, j]), ha = "center", va = "center", color = color, fontsize = 20, fontweight = "bold")

    # Accuracy & Info Table
    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.axis('off')
    total_params = sum(w.size for w in model.weights) + sum(b.size for b in model.biases) if model.use_bias else sum(w.size for w in model.weights)
    table_data = [
        ["Hidden Layers", str(model.num_hidden_layers)],
        ["Neurons per Layer", str(model.neurons_per_layer)],
        ["Activation", model.activation_function.capitalize()],
        ["Learning Rate", f"{model.learning_rate:.4f}"],
        ["Epochs", str(model.n_epochs)],
        ["Bias Used", "Yes" if model.use_bias else "No"],
        ["Total Parameters", str(total_params)],
        ["Final MSE", f"{model.errors_history[-1]:.6f}"],
        ["Test Accuracy", f"{accuracy_value:.2f}%"],
        ["Classes", "Adelie, Chinstrap, Gentoo"],
    ]
    table = ax3.table(cellText = table_data,
                      colLabels = ["Parameter", "Value"],
                      cellLoc = 'left',
                      loc = 'center',
                      colWidths = [0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.8)
    for i in range(2): # Header
        table[(0, i)].set_facecolor('#0EA5E9')
        table[(0, i)].set_text_props(weight = 'bold', color = 'white')
    # Highlight accuracy row
    table[(8, 0)].set_facecolor('#E8F5E8')
    table[(8, 1)].set_facecolor('#E8F5E8')
    # Sample Predictions
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    # Show 10 random predictions
    indices = random.sample(range(len(X_test)), 10)
    sample_data = []
    for idx in indices:
        true = [class1, class2, class3][y_test[idx]]
        pred = [class1, class2, class3][y_pred[idx]]
        status = "Correct" if true == pred else "Wrong"
        sample_data.append([f"Sample {idx + 1}", true, pred, status])

    sample_table = ax4.table(cellText = sample_data,
                             colLabels = ["Sample", "True", "Predicted", "Result"],
                             cellLoc = 'center',
                             loc = 'center')
    sample_table.auto_set_font_size(False)
    sample_table.set_fontsize(11)
    sample_table.scale(1, 2.2)

    # alwan el correct wl wrong
    for i in range(1, 11):
        if sample_data[i - 1][3] == "Correct":
            sample_table[(i, 3)].set_facecolor('#C8E6C9')
        else:
            sample_table[(i, 3)].set_facecolor('#FFCDD2')

    plt.suptitle("Penguin Species Classification - Backpropagation Results",
                 fontsize = 20, fontweight = 'bold', y = 0.98)

    plt.tight_layout()

    # Saving the plots
    filename = f"penguin_backprop_result_{random.randint(1000, 9999)}.png"
    filepath = os.path.join("Visualizations", filename)
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight', facecolor = 'white')
    plt.close()

    print(f"Saved: {filepath}")
    return filepath