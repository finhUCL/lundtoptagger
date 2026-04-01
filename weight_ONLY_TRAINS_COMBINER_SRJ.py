import argparse
from datetime import datetime
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import uproot

from tools.GNN_model_weight.utils_newdata import load_yaml
from tools.utils_config import recursive_update, parse_dot_args

from tools.GNN_model_weight.models import Combiner, CombinerV2


def build_combiner(model_cfg):
    name = model_cfg.get("name", "Combiner")
    if name == "Combiner":
        return Combiner()
    if name == "CombinerV2":
        hidden_dims = tuple(model_cfg.get("hidden_dims", [64, 32, 16]))
        dropout = float(model_cfg.get("dropout", 0.2))
        n_kinematics = int(model_cfg.get("n_kinematics", 0))
        return CombinerV2(hidden_dims=hidden_dims, dropout=dropout, n_kinematics=n_kinematics)
    raise ValueError(f"Unknown combiner model name: '{name}'. Choose 'Combiner' or 'CombinerV2'.")


def resolve_root_files(root_dir, root_pattern="*.root"):
    if os.path.isfile(root_dir):
        return [root_dir]

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Input path does not exist: {root_dir}")

    files = sorted(glob.glob(os.path.join(root_dir, root_pattern)))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {root_dir} matching pattern: {root_pattern}")

    return files


def extract_from_root(files, tree_name, label_branch, score_branches, kinematic_branches=None):
    if len(score_branches) != 2:
        raise ValueError("score_branches must contain exactly two branch names.")

    score1_branch, score2_branch = score_branches
    kin_branches = kinematic_branches or []
    all_branches = [label_branch, score1_branch, score2_branch] + kin_branches

    labels_all, score1_all, score2_all = [], [], []
    kin_all = [[] for _ in kin_branches]

    for file_path in files:
        with uproot.open(file_path) as f_in:
            if tree_name not in f_in:
                raise KeyError(f"Tree '{tree_name}' not found in {file_path}")

            tree = f_in[tree_name]
            arrays = tree.arrays(all_branches, library="np")

        labels_all.append(np.asarray(arrays[label_branch]).reshape(-1))
        score1_all.append(np.asarray(arrays[score1_branch]).reshape(-1))
        score2_all.append(np.asarray(arrays[score2_branch]).reshape(-1))
        for i, b in enumerate(kin_branches):
            kin_all[i].append(np.asarray(arrays[b]).reshape(-1))

    labels = np.concatenate(labels_all, axis=0)
    score1 = np.concatenate(score1_all, axis=0)
    score2 = np.concatenate(score2_all, axis=0)
    kinematics = [np.concatenate(kin_all[i], axis=0) for i in range(len(kin_branches))]
    return labels, score1, score2, kinematics


def preprocess_for_training(labels, score1, score2, kinematics=None):
    labels = np.asarray(labels).reshape(-1)
    score1 = np.asarray(score1).reshape(-1)
    score2 = np.asarray(score2).reshape(-1)
    kin_arrays = [np.asarray(k).reshape(-1) for k in (kinematics or [])]

    valid_mask = np.isfinite(labels) & np.isfinite(score1) & np.isfinite(score2)
    for k in kin_arrays:
        valid_mask &= np.isfinite(k)
    labels = labels[valid_mask]
    score1 = score1[valid_mask]
    score2 = score2[valid_mask]
    kin_arrays = [k[valid_mask] for k in kin_arrays]

    labels = np.rint(labels).astype(np.int64)
    binary_mask = (labels == 0) | (labels == 1)
    labels = labels[binary_mask]
    score1 = score1[binary_mask]
    score2 = score2[binary_mask]
    kin_arrays = [k[binary_mask] for k in kin_arrays]

    if labels.size == 0:
        raise ValueError("No valid binary labels found after preprocessing.")

    cols = [score1, score2] + kin_arrays
    x = np.stack(cols, axis=1).astype(np.float32)
    y = labels.astype(np.float32).reshape(-1, 1)
    return x, y


def compute_kin_normalisation(x_train, n_scores=2):
    """Compute mean and std for kinematic columns (columns 2+) from training data."""
    if x_train.shape[1] <= n_scores:
        return np.array([]), np.array([])
    kin = x_train[:, n_scores:]
    mean = kin.mean(axis=0)
    std = kin.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_kin_normalisation(x, mean, std, n_scores=2):
    """Z-score normalise kinematic columns in-place (returns a copy)."""
    if mean.size == 0:
        return x
    x = x.copy()
    x[:, n_scores:] = (x[:, n_scores:] - mean) / std
    return x


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

    kin_cfg = config.get("kinematics", {})
    kinematic_branches = kin_cfg.get("branches", [])

    validation_fraction = float(train_cfg["validation_fraction"])
    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        raise ValueError("training.validation_fraction must be between 0 and 1.")

    n_epochs = int(train_cfg["n_epochs"])
    batch_size = int(train_cfg["batch_size"])
    learning_rate = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    early_stopping_patience = train_cfg.get("early_stopping_patience", None)
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
    lr_scheduler_cfg = train_cfg.get("lr_scheduler", None)
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

    # Derive n_kinematics from config so model matches data
    model_cfg = config.get("model", {})
    if kinematic_branches:
        model_cfg["n_kinematics"] = len(kinematic_branches)
    else:
        model_cfg["n_kinematics"] = 0

    files = resolve_root_files(root_dir, root_pattern)
    print(f"Found {len(files)} ROOT file(s) to load.")

    if kinematic_branches:
        print(f"Loading kinematic branches: {kinematic_branches}")

    labels, score1, score2, kinematics = extract_from_root(
        files, tree_name, label_branch, score_branches, kinematic_branches
    )
    x, y = preprocess_for_training(labels, score1, score2, kinematics)
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

    # Normalise kinematic columns using training-set statistics
    kin_mean, kin_std = compute_kin_normalisation(x_train, n_scores=2)
    x_train = apply_kin_normalisation(x_train, kin_mean, kin_std, n_scores=2)
    x_val   = apply_kin_normalisation(x_val,   kin_mean, kin_std, n_scores=2)

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

    # Save normalisation stats so the scoring script can reuse them
    if kin_mean.size > 0:
        norm_stats = {
            "kinematic_branches": kinematic_branches,
            "mean": kin_mean.tolist(),
            "std":  kin_std.tolist(),
        }
        norm_path = os.path.join(save_dir, "kin_normalisation.json")
        with open(norm_path, "w", encoding="utf-8") as f_norm:
            json.dump(norm_stats, f_norm, indent=2)
        print(f"Saved kinematic normalisation stats to: {norm_path}")

    model = build_combiner(model_cfg).to(device)
    print(f"Model: {model_cfg.get('name', 'Combiner')}  |  parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    scheduler = None
    if lr_scheduler_cfg is not None:
        sched_type = lr_scheduler_cfg.get("type", "plateau")
        if sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(lr_scheduler_cfg.get("factor", 0.5)),
                patience=int(lr_scheduler_cfg.get("patience", 5)),
                min_lr=float(lr_scheduler_cfg.get("min_lr", 1e-6)),
            )
        elif sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=n_epochs,
                eta_min=float(lr_scheduler_cfg.get("min_lr", 1e-6)),
            )
        else:
            raise ValueError(f"Unknown lr_scheduler type: '{sched_type}'. Choose 'plateau' or 'cosine'.")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    val_loss_path = os.path.join(save_dir, val_loss_filename)
    with open(val_loss_path, "w", encoding="utf-8") as f_out:
        f_out.write("epoch,train_loss,val_loss,lr\n")

    best_ckpt_path = None

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
        current_lr = optimizer.param_groups[0]["lr"]

        with open(val_loss_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{current_lr:.2e}\n")

        print(
            f"Epoch {epoch:03d}/{n_epochs:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | lr={current_lr:.2e}"
        )

        if save_every_epoch or epoch == n_epochs:
            ckpt_name = f"{checkpoint_prefix}_e{epoch:03d}_{val_loss:.5f}.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))

        # Always persist the best checkpoint separately for easy retrieval
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_ckpt_path = os.path.join(save_dir, f"{checkpoint_prefix}_best.pt")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> New best val_loss={best_val_loss:.6f}  (saved {checkpoint_prefix}_best.pt)")
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping: no improvement for {early_stopping_patience} epochs.")
            break

    print(f"Training complete. Checkpoints and losses saved to: {save_dir}")
    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}  (val_loss={best_val_loss:.6f})")


if __name__ == "__main__":
    main()
