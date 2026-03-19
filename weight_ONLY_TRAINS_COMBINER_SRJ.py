import argparse
from datetime import datetime
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import uproot

from tools.GNN_model_weight.utils_newdata import load_yaml
from tools.utils_config import recursive_update, parse_dot_args

try:
    from tools.GNN_model_weight.models import Combiner as CombinerModel
except ImportError:
    from tools.GNN_model_weight.models import combiner as CombinerModel


def resolve_root_files(root_dir, root_pattern="*.root"):
    if os.path.isfile(root_dir):
        return [root_dir]

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Input path does not exist: {root_dir}")

    files = sorted(glob.glob(os.path.join(root_dir, root_pattern)))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {root_dir} matching pattern: {root_pattern}")

    return files


def extract_from_root(files, tree_name, label_branch, score_branches):
    if len(score_branches) != 2:
        raise ValueError("score_branches must contain exactly two branch names.")

    score1_branch, score2_branch = score_branches
    labels_all, score1_all, score2_all = [], [], []

    for file_path in files:
        with uproot.open(file_path) as f_in:
            if tree_name not in f_in:
                raise KeyError(f"Tree '{tree_name}' not found in {file_path}")

            tree = f_in[tree_name]
            arrays = tree.arrays([label_branch, score1_branch, score2_branch], library="np")

        labels_all.append(np.asarray(arrays[label_branch]).reshape(-1))
        score1_all.append(np.asarray(arrays[score1_branch]).reshape(-1))
        score2_all.append(np.asarray(arrays[score2_branch]).reshape(-1))

    labels = np.concatenate(labels_all, axis=0)
    score1 = np.concatenate(score1_all, axis=0)
    score2 = np.concatenate(score2_all, axis=0)
    return labels, score1, score2


def preprocess_for_training(labels, score1, score2):
    labels = np.asarray(labels).reshape(-1)
    score1 = np.asarray(score1).reshape(-1)
    score2 = np.asarray(score2).reshape(-1)

    valid_mask = np.isfinite(labels) & np.isfinite(score1) & np.isfinite(score2)
    labels = labels[valid_mask]
    score1 = score1[valid_mask]
    score2 = score2[valid_mask]

    labels = np.rint(labels).astype(np.int64)
    binary_mask = (labels == 0) | (labels == 1)
    labels = labels[binary_mask]
    score1 = score1[binary_mask]
    score2 = score2[binary_mask]

    if labels.size == 0:
        raise ValueError("No valid binary labels found after preprocessing.")

    x = np.stack((score1, score2), axis=1).astype(np.float32)
    y = labels.astype(np.float32).reshape(-1, 1)
    return x, y


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            total_count += batch_count

    return total_loss / max(total_count, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Combiner model from ROOT score branches")
    parser.add_argument("config", help="path to yaml config")
    parser.add_argument("--override", nargs='*', default=[])
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.override:
        config = recursive_update(config, parse_dot_args(args.override))

    data_cfg = config["data"]
    train_cfg = config["training"]
    output_cfg = config["output"]

    root_dir = data_cfg["root_dir"]
    root_pattern = data_cfg.get("root_pattern", "*.root")
    tree_name = data_cfg.get("tree_name", "FlatSubstructureJetTree")
    label_branch = data_cfg["label_branch"]
    score_branches = data_cfg["score_branches"]

    validation_fraction = float(train_cfg["validation_fraction"])
    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        raise ValueError("training.validation_fraction must be between 0 and 1.")

    n_epochs = int(train_cfg["n_epochs"])
    batch_size = int(train_cfg["batch_size"])
    learning_rate = float(train_cfg["learning_rate"])
    num_workers = int(config.get("num_workers", 0))
    seed = int(config.get("seed", 42))

    save_every_epoch = bool(output_cfg.get("save_every_epoch", True))
    checkpoint_prefix = output_cfg.get("checkpoint_prefix", "Combiner")
    path_to_save = output_cfg["path_to_save"]
    val_loss_filename = output_cfg.get("val_loss_filename", "validation_losses.txt")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(path_to_save, f"{checkpoint_prefix}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    files = resolve_root_files(root_dir, root_pattern)
    print(f"Found {len(files)} ROOT file(s) to load.")

    labels, score1, score2 = extract_from_root(files, tree_name, label_branch, score_branches)
    x, y = preprocess_for_training(labels, score1, score2)
    print(f"Loaded {len(y)} jets after preprocessing.")

    y_flat = y.reshape(-1)
    unique_classes = np.unique(y_flat)
    stratify = y_flat if unique_classes.size > 1 else None

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=validation_fraction,
        random_state=seed,
        stratify=stratify,
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if torch.cuda.is_available():
        gpu_cfg = config.get("gpu", None)
        device_id = "cuda" if gpu_cfg is None else f"cuda:{gpu_cfg}"
    else:
        device_id = "cpu"
    device = torch.device(device_id)
    print(f"Using device: {device}")

    model = CombinerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    val_loss_path = os.path.join(save_dir, val_loss_filename)
    with open(val_loss_path, "w", encoding="utf-8") as f_out:
        f_out.write("epoch,train_loss,val_loss\n")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_count = 0

        for features, labels_batch in train_loader:
            features = features.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            batch_count = labels_batch.size(0)
            total_train_loss += loss.item() * batch_count
            total_train_count += batch_count

        train_loss = total_train_loss / max(total_train_count, 1)
        val_loss = evaluate(model, val_loader, criterion, device)

        with open(val_loss_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")

        print(f"Epoch {epoch:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if save_every_epoch or epoch == n_epochs:
            ckpt_name = f"{checkpoint_prefix}_e{epoch:03d}_{val_loss:.5f}.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))

    print(f"Training complete. Checkpoints and losses saved to: {save_dir}")


if __name__ == "__main__":
    main()
