import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def visualize_data(feature1, feature2, class1, class2, learning_rate, n_epochs, mse_threshold,
                   add_bias, algorithm, X_train, y_train, X_test, y_test, y_pred, weights, bias_val):
    """
    Visualize training results including decision boundary, confusion matrix, and summary table.
    Saves the plot as a PNG file in the 'visualizations' directory.
    Args:
        feature1: Name of the first feature
        feature2: Name of the second feature
        class1: Name of the first class
        class2: Name of the second class
        learning_rate: Learning rate used in training
        n_epochs: Number of epochs used in training
        mse_threshold: MSE threshold for Adaline (None for Perceptron)
        add_bias: Boolean indicating if bias was included
        algorithm: Name of the algorithm used ("Perceptron" or "Adaline")
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        y_pred: Predicted labels for test set
        weights: Weights learned by the model
        bias_val: Bias learned by the model
    """

    # Normalize inputs
    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Convert labels to binary (0 and 1)
    y_train = np.where(y_train <= 0, 0, 1)
    y_test = np.where(y_test <= 0, 0, 1)
    y_pred = np.where(y_pred <= 0, 0, 1)

    # Handle weights shape
    weights = np.array(weights, dtype=float).flatten()
    if len(weights) == 3 and add_bias:
        b = weights[0]
        w1, w2 = weights[1], weights[2]
    elif len(weights) >= 2:
        w1, w2 = weights[0], weights[1]
        b = bias_val if add_bias else 0

    # Calculate accuracy and confusion matrix
    accuracy_value = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Figure setup
    fig = plt.figure(figsize=(17, 6))
    gs = GridSpec(1, 3, width_ratios=[2.2, 1, 1.2], figure=fig)

    # Scatter Plot
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
                color='blue', label=f'{class1} (-1)')
    ax1.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
                color='orange', label=f'{class2} (+1)')

    # Decision boundary
    x_values = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
    if w2 != 0:
        y_values = -(b + w1 * x_values) / w2
        ax1.plot(x_values, y_values, color='green', label='Decision Boundary')
    else:
        ax1.axvline(-b / w1 if w1 != 0 else 0, color='green', linestyle='--', label='Decision Boundary')

    ax1.set_title(f"Test Data — Decision Boundary\n(+1 → {class2},  -1 → {class1})", fontsize=12)
    ax1.set_xlabel(feature1)
    ax1.set_ylabel(feature2)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Confusion Matrix Plotting
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title("Confusion Matrix", fontsize=12)
    ax2.set_xticks(np.arange(2))
    ax2.set_xticklabels([f'Pred {class2}', f'Pred {class1}'])
    ax2.set_yticks(np.arange(2))
    ax2.set_yticklabels([f'Actual {class2}', f'Actual {class1}'])

    labels = np.array([["TP", "FN"], ["FP", "TN"]])
    thresh = cm.max() / 2. if cm.max() != 0 else 0
    for i, j in np.ndindex(cm.shape):
        color = "white" if cm[i, j] > thresh else "black"
        ax2.text(j, i - 0.15, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
        ax2.text(j, i + 0.35, f"{labels[i, j]}", ha="center", va="center", color=color, fontsize=9)

    ax2.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax2.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax2.grid(which="minor", color="gray", linestyle='--', linewidth=0.5)
    ax2.tick_params(which="minor", bottom=False, left=False)

    # Combinations Table
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    table_data = [
        ["Features Used", f"{feature1}, {feature2}"],
        ["Classes Used", f"{class1}, {class2}"],
        ["Learning Rate", f"{learning_rate}"],
        ["Epochs", f"{n_epochs}"],
        ["Bias Included", "Yes" if add_bias else "No"],
        ["Bias Value", f"{b:.4f}" if add_bias or b != 0 else "—"],
        ["Weights", f"[{w1:.4f}, {w2:.4f}]"],
        ["Accuracy", f"{accuracy_value*100:.2f}%"],
        ["Algorithm", algorithm],
    ]
    if algorithm.lower() == "adaline":
        table_data.append(["MSE Threshold", f"{mse_threshold}"])

    table = ax3.table(cellText=table_data,
                      colLabels=["Parameter", "Value"],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.45, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.4)

    # Saving the plot
    plt.tight_layout()
    filename = f"training_visualization_{random.randint(1,9999)}.png"
    filepath = f"./visualizations/{filename}"
    plt.savefig(filepath, bbox_inches="tight")
    print(f"✅ Plot saved as {filepath}")


# Helper Functions

def accuracy(y_test, y_pred):
    """
    Calculate accuracy of predictions.
    Args:
    y_test: Test labels
    y_pred: Predicted labels for test set
    """
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sum(y_test == y_pred) / len(y_test)


def confusion_matrix(y_test, y_pred):
    """
    Compute confusion matrix.
    Args:
    y_test: Test labels
    y_pred: Predicted labels for test set
    """
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    y_test = np.where(y_test <= 0, 0, 1)
    y_pred = np.where(y_pred <= 0, 0, 1)
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    return np.array([[tp, fn], [fp, tn]])
