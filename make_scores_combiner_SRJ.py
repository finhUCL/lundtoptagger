import argparse
import gc
import glob
import json
import os
import time

import numpy as np
import torch
import uproot
from torch_geometric.loader import DataLoader

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import load_yaml, get_scores
from tools.utils_config import recursive_update, parse_dot_args

from tools.GNN_model_weight.models import Combiner, CombinerV2


def build_combiner_by_name(name, model_cfg=None):
    if model_cfg is None:
        model_cfg = {}
    if name == "Combiner":
        return Combiner()
    if name == "CombinerV2":
        hidden_dims = tuple(model_cfg.get("hidden_dims", [64, 32, 16]))
        dropout = float(model_cfg.get("dropout", 0.2))
        n_kinematics = int(model_cfg.get("n_kinematics", 0))
        return CombinerV2(hidden_dims=hidden_dims, dropout=dropout, n_kinematics=n_kinematics)
    raise ValueError(f"Unknown combiner model name: '{name}'. Choose 'Combiner' or 'CombinerV2'.")


PART_SCORE_BRANCH = "parT_score"


def build_model_by_name(name):
    if name == "LundNet":
        return LundNet()
    elif name == "GATNet":
        return GATNet()
    elif name == "GINNet":
        return GINNet()
    elif name == "EdgeGinNet":
        return EdgeGinNet()
    elif name == "PNANet":
        return PNANet()
    elif name == "LundNet_plus_GN2X":
        return LundNet_plus_GN2X()
    elif name == "LundNet_plus_GN3X":
        return LundNet_plus_GN3X()
    elif name == "QLundNet":
        return QLundNet()
    else:
        raise ValueError(f"Unknown model type {name}")


def merge_root_files(file_list, output_name, tree_name):
    if not file_list:
        return

    print(f"  -> Collecting all data for merge: {os.path.basename(output_name)}")
    all_data = []
    for i, file_path in enumerate(file_list):
        with uproot.open(file_path) as f_in:
            all_data.append(f_in[tree_name].arrays())
            print(f"     Read shard {i + 1}/{len(file_list)}")

    import awkward as ak

    print("  -> Concatenating arrays...")
    combined_data = ak.concatenate(all_data)

    print(f"  -> Writing to {os.path.basename(output_name)}...")
    with uproot.recreate(output_name) as f_out:
        f_out[tree_name] = combined_data

    print("  -> Merge complete.")


def main():
    parser = argparse.ArgumentParser(description="Run SRJ scoring with LundNet + Combiner")
    parser.add_argument("config", help="job configuration")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    config = load_yaml(args.config)
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    kT_selection = config["data"]["kT_cut"]
    placeholder = {
        "sample": config["data"]["sample"],
        "kT_cut": kT_selection,
    }

    paths_graphs = config["data"]["paths_to_test_file_graphs"]
    if isinstance(paths_graphs, str):
        paths_graphs = [paths_graphs]
    files_graphs = []
    for pattern in paths_graphs:
        files_graphs.extend(glob.glob(pattern.format(**placeholder)))
    files_graphs.sort()
    if not files_graphs:
        raise FileNotFoundError("No graph input files matched paths_to_test_file_graphs.")

    paths_root = config["data"]["paths_to_test_file_root"]
    if isinstance(paths_root, str):
        paths_root = [paths_root]
    files_root = []
    for pattern in paths_root:
        files_root.extend(glob.glob(pattern.format(**placeholder)))
    files_root.sort()
    if not files_root:
        raise FileNotFoundError("No ROOT input files matched paths_to_test_file_root.")

    print(f"Matched {len(files_graphs)} graph files and {len(files_root)} ROOT files.")

    if len(files_graphs) != len(files_root):
        raise ValueError(f"Mismatch: {len(files_graphs)} graphs vs {len(files_root)} roots!")

    outdir = config["data"]["path_to_outdir"].format(**placeholder)
    os.makedirs(outdir, exist_ok=True)
    output_suffix = config["data"]["output_suffix"].format(**placeholder)

    tree_name = config["data"].get("tree_name", "FlatSubstructureJetTree")
    batch_size = config["test"]["batch_size"]
    score_branch_template = config["test"].get("scores_branch_name", "fjet_{tag}_score")

    lund_tag = config["test"]["lundnet_model"]["tag"]
    lund_arch = config["test"]["lundnet_model"]["arch"]
    lund_ckpt = config["test"]["lundnet_model"]["ckpt"]

    combined_tag = config["test"]["combiner_model"]["tag"]
    combiner_ckpt = config["test"]["combiner_model"]["ckpt"]

    lund_score_branch = score_branch_template.format(tag=lund_tag)
    combined_score_branch = score_branch_template.format(tag=combined_tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading LundNet model ({lund_arch}) checkpoint...")
    lund_model = build_model_by_name(lund_arch)
    lund_model.load_state_dict(torch.load(lund_ckpt, map_location=device))
    lund_model.to(device)
    lund_model.eval()

    combiner_name = config["test"]["combiner_model"].get("name", "Combiner")
    combiner_model_cfg = config["test"]["combiner_model"]
    print(f"Loading {combiner_name} checkpoint...")
    combiner_model = build_combiner_by_name(combiner_name, combiner_model_cfg)
    combiner_model.load_state_dict(torch.load(combiner_ckpt, map_location=device))
    combiner_model.to(device)
    combiner_model.eval()

    # Load kinematic normalisation stats if provided
    kin_norm_path = combiner_model_cfg.get("kin_normalisation", None)
    kinematic_branches = combiner_model_cfg.get("kinematic_branches", [])
    kin_mean = kin_std = None
    if kin_norm_path:
        with open(kin_norm_path, "r", encoding="utf-8") as f_norm:
            norm_stats = json.load(f_norm)
        kin_mean = np.array(norm_stats["mean"], dtype=np.float32)
        kin_std  = np.array(norm_stats["std"],  dtype=np.float32)
        print(f"Loaded kinematic normalisation for: {norm_stats['kinematic_branches']}")

    print(f"LundNet score branch: {lund_score_branch}")
    print(f"Combined score branch: {combined_score_branch}")

    generated_shards = []
    t_global = time.time()

    for file_number, (fg, fr) in enumerate(zip(files_graphs, files_root), start=1):
        print(f"\n[{file_number}/{len(files_graphs)}] Processing shard: {os.path.basename(fg)}")

        dataset = torch.load(fg, weights_only=False)
        n_jets = len(dataset)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        lund_pred = get_scores(test_loader, lund_model, device)
        lund_scores = np.asarray(lund_pred[:, 0]).reshape(-1).astype(np.float32)

        shard_out_name = os.path.join(outdir, os.path.basename(fr).replace(".root", f"{output_suffix}.root"))

        with uproot.open(fr) as f_in:
            if tree_name not in f_in:
                raise KeyError(f"Tree '{tree_name}' not found in {fr}")
            arrays = f_in[tree_name].arrays()

        if len(arrays) != n_jets:
            raise ValueError(
                f"Entry mismatch for {os.path.basename(fg)}: ROOT has {len(arrays)} entries, graph has {n_jets}."
            )

        if PART_SCORE_BRANCH not in arrays.fields:
            raise KeyError(f"Branch '{PART_SCORE_BRANCH}' not found in ROOT file {fr}")

        part_scores = np.asarray(arrays[PART_SCORE_BRANCH]).reshape(-1).astype(np.float32)
        if part_scores.shape[0] != n_jets:
            raise ValueError(
                f"Branch '{PART_SCORE_BRANCH}' has {part_scores.shape[0]} entries, expected {n_jets}."
            )

        feature_cols = [lund_scores, part_scores]
        if kinematic_branches:
            for branch in kinematic_branches:
                if branch not in arrays.fields:
                    raise KeyError(f"Kinematic branch '{branch}' not found in ROOT file {fr}")
                feature_cols.append(np.asarray(arrays[branch]).reshape(-1).astype(np.float32))

        feature_arr = np.stack(feature_cols, axis=1)

        if kin_mean is not None:
            feature_arr[:, 2:] = (feature_arr[:, 2:] - kin_mean) / kin_std

        features = torch.from_numpy(feature_arr).to(device)
        with torch.no_grad():
            combined_scores = combiner_model(features).cpu().numpy().reshape(-1).astype(np.float32)

        arrays[lund_score_branch] = lund_scores
        arrays[combined_score_branch] = combined_scores

        print(f"  Saving shard to: {os.path.basename(shard_out_name)}")
        with uproot.recreate(shard_out_name) as f_out:
            f_out[tree_name] = arrays

        generated_shards.append(shard_out_name)

        del dataset, test_loader, lund_pred, arrays, features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if generated_shards:
        final_outfile = os.path.join(outdir, f"Final_Scores_Combined_{config['data']['sample']}_kT{kT_selection}.root")
        print(f"\n=== Merging {len(generated_shards)} shards into single file ===")

        try:
            merge_root_files(generated_shards, final_outfile, tree_name)
            print("Successfully created combined ROOT file.")

            print(f"=== Cleaning up {len(generated_shards)} temporary shards ===")
            for shard_path in generated_shards:
                try:
                    if os.path.exists(shard_path):
                        os.remove(shard_path)
                except Exception as e:
                    print(f"Warning: Failed to delete {shard_path}: {e}")
            print("Cleanup complete.")

        except Exception as e:
            print(f"Error during Python merging: {e}")
            print("Cleanup aborted to preserve shards for debugging.")

    t_total = time.time() - t_global
    m, s = divmod(int(t_total), 60)
    print(f"\nTotal evaluation and merging time: {m} min {s} s")


if __name__ == "__main__":
    main()
