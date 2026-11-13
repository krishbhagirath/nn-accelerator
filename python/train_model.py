# Trains small fully-connected networks and exports them to ONNX.
#
# This file ONLY does float work:
#   - load a dataset
#   - train an MLP (Linear -> ReLU -> ... -> Linear)
#   - export it to ONNX
#   - save calibration / test inputs as text
#
# All quantization lives in compile_onnx.py, so there is one source of truth.
#
# Usage:  python python/train_model.py --model digits
#         python python/train_model.py --model all

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# torch.onnx prints emoji; the default Windows console encoding can't print them.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Every model the project supports. hidden = sizes of the hidden ReLU layers.
# All sizes must fit the RTL limits (MAX_INPUTS=64, MAX_OUTPUTS=32).
MODEL_CONFIGS = {
    "tiny":        {"dataset": "synthetic",     "hidden": [16],     "epochs": 100},
    "iris":        {"dataset": "iris",          "hidden": [8],      "epochs": 300},
    "cancer":      {"dataset": "breast_cancer", "hidden": [16],     "epochs": 200},
    "digits":      {"dataset": "digits",        "hidden": [16],     "epochs": 300},
    "digits_deep": {"dataset": "digits",        "hidden": [32, 16], "epochs": 300},
}


# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------

def load_synthetic():
    """The project's original 8-feature, 2-class toy problem (kept identical)."""
    X = torch.randn(2000, 8)
    score = (2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + 1.2 * X[:, 3]
             - 0.5 * X[:, 4] + 0.7 * X[:, 5] - 1.0 * X[:, 6] + 0.4 * X[:, 7])
    y = (score > 0).long()
    X, y = X.numpy(), y.numpy()
    return X[:1600], y[:1600], X[1600:], y[1600:]


def load_sklearn(name):
    if name == "iris":
        data = datasets.load_iris()
    elif name == "breast_cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_digits()

    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)

    if name == "digits":
        # Pixels are 0..16. Dividing by 16 maps them to 0..1, like an image.
        X_train, X_test = X_train / 16.0, X_test / 16.0
    else:
        # Feature columns have very different units (e.g. mm vs mm^2), so we
        # standardize them using TRAINING statistics. This preprocessing happens
        # in software before the accelerator, like image normalization would.
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    return (X_train.astype(np.float32), y_train,
            X_test.astype(np.float32), y_test)


def load_dataset(name):
    if name == "synthetic":
        return load_synthetic()
    return load_sklearn(name)


# ------------------------------------------------------------
# Model + training
# ------------------------------------------------------------

def build_mlp(input_dim, hidden_sizes, num_classes):
    """Linear -> ReLU -> ... -> Linear. Exactly the ops the accelerator supports."""
    layers = []
    previous = input_dim
    for size in hidden_sizes:
        layers += [nn.Linear(previous, size), nn.ReLU()]
        previous = size
    layers.append(nn.Linear(previous, num_classes))
    return nn.Sequential(*layers)


def train(model, X_train, y_train, epochs):
    # Full-batch Adam: the datasets are tiny, so this is simple and fast.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.from_numpy(X_train)
    y = torch.from_numpy(y_train)
    for epoch in range(epochs):
        loss = loss_fn(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def accuracy(model, X, y):
    with torch.no_grad():
        predictions = model(torch.from_numpy(X)).argmax(dim=1).numpy()
    return float((predictions == y).mean())


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------

def export_model(name):
    config = MODEL_CONFIGS[name]

    # Seed first, then draw the data, then build the model: this ordering keeps
    # the "tiny" model identical to the original project's model.
    torch.manual_seed(42)
    X_train, y_train, X_test, y_test = load_dataset(config["dataset"])

    num_classes = int(max(y_train.max(), y_test.max())) + 1
    model = build_mlp(X_train.shape[1], config["hidden"], num_classes)
    final_loss = train(model, X_train, y_train, config["epochs"])
    test_accuracy = accuracy(model, X_test, y_test)

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (torch.from_numpy(X_test[:1]),),
        out_dir / "model.onnx",
        input_names=["input"],
        output_names=["output"],
        external_data=False,
        verbose=False,
    )

    # Text files: one sample per line, features separated by spaces.
    np.savetxt(out_dir / "calibration_inputs.txt", X_train, fmt="%.8g")
    np.savetxt(out_dir / "test_inputs.txt", X_test, fmt="%.8g")
    np.savetxt(out_dir / "test_labels.txt", y_test, fmt="%d")

    dims = [X_train.shape[1]] + config["hidden"] + [num_classes]
    info = {
        "model": name,
        "dataset": config["dataset"],
        "architecture": " -> ".join(str(d) for d in dims),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "torch_float_test_accuracy": test_accuracy,
        "final_train_loss": final_loss,
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=4))

    print(f"[train] {name:12s} {info['architecture']:16s} "
          f"params={info['num_parameters']:5d}  float test acc={test_accuracy * 100:.2f}%")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all"] + list(MODEL_CONFIGS))
    args = parser.parse_args()
    names = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    for name in names:
        export_model(name)


if __name__ == "__main__":
    main()
