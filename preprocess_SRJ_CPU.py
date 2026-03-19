import os
import json
import torch
import numpy as np
import uproot
import glob
import matplotlib.pyplot as plt

from tools.GNN_model_weight.utils_newdata import load_yaml, assign_flat_weights, assign_2d_flat_weights_kde
from plotting.utils_plots_matplotlib import hist_with_errors

# ============================================================
# 1. Plotting Helper Function (Branching Logic: 1D vs 2D)
# ============================================================
def _save_plots(pts, etas, labels, weights_sig, weights_bkg, config, out_dir):
    plot_dir = os.path.join(out_dir, "plots_validation")
    os.makedirs(plot_dir, exist_ok=True)

    sig_mask, bkg_mask = (labels == 1), (labels == 0)
    pts_sig, pts_bkg = pts[sig_mask], pts[bkg_mask]
    etas_sig, etas_bkg = etas[sig_mask], etas[bkg_mask]
    
    do_pt, do_eta = config.get("flatten_pt", False), config.get("flatten_eta", False)

    if do_pt and not do_eta:
        print("\n[Check] Plotting 1D validation (pT only)...")
        plt.figure(figsize=(10, 6))
        hist_with_errors(pts_bkg, bins=config["n_bins_pt"], density=True, label="Bkg (Before)")
        hist_with_errors(pts_sig, bins=config["n_bins_pt"], density=True, label="Sgn (Before)")
        plt.xlabel("SRJ pT [GeV]"); plt.ylabel("Density"); plt.legend(); plt.savefig(os.path.join(plot_dir, "pT_1D_before.png")); plt.close()

        plt.figure(figsize=(10, 6))
        hist_with_errors(pts_bkg, weights=weights_bkg, bins=config["n_bins_pt"], density=True, label="Bkg (After)")
        hist_with_errors(pts_sig, weights=weights_sig, bins=config["n_bins_pt"], density=True, label="Sgn (After)")
        plt.xlabel("SRJ pT [GeV]"); plt.ylabel("Density"); plt.legend(); plt.savefig(os.path.join(plot_dir, "pT_1D_after.png")); plt.close()

    elif do_pt and do_eta:
        print("\n[Check] Plotting 2D validation (pT & eta)...")
        def plot_2d_mesh(p, e, w, title, filename):
            plt.figure(figsize=(8, 6))
            plt.hist2d(p, e, bins=(config["n_bins_pt"], config["n_bins_eta"]), weights=w, density=True, cmap='viridis', cmin=1e-12)
            plt.colorbar(label="Density"); plt.xlabel("SRJ pT [GeV]"); plt.ylabel("SRJ η"); plt.title(title)
            plt.savefig(os.path.join(plot_dir, filename)); plt.close()

        plot_2d_mesh(pts_sig, etas_sig, None, "Signal 2D: Before", "2D_sig_before.png")
        plot_2d_mesh(pts_sig, etas_sig, weights_sig, "Signal 2D: After 2D Flatten", "2D_sig_after.png")
        plot_2d_mesh(pts_bkg, etas_bkg, None, "Background 2D: Before", "2D_bkg_before.png")
        plot_2d_mesh(pts_bkg, etas_bkg, weights_bkg, "Background 2D: After 2D Flatten", "2D_bkg_after.png")

    print(f"[Check] Validation plots saved to: {plot_dir}")

# ============================================================
# 2. Core Processing Function
# ============================================================
def process_one_split(graph_paths, config, mean_x, std_x, mean_ntrk, std_ntrk, do_flatten, tag, out_dir):
    print(f"\n{'='*70}\n>>>>>> Starting Processing for [{tag.upper()}] <<<<<<\n{'='*70}")
    os.makedirs(out_dir, exist_ok=True)
    
    all_graph_files = sorted(glob.glob(graph_paths[0]) if isinstance(graph_paths, list) and "*" in graph_paths[0] else graph_paths)
    all_root_files = [os.path.join(os.path.dirname(g_p), os.path.basename(g_p).replace("graphs_", "data_").replace("_with_pt", "") + ".root") for g_p in all_graph_files]

    all_weights_sig, all_weights_bkg = None, None
    pts, etas, labels = None, None, None

    # Stage 1: Scan ROOT
    if do_flatten:
        print(f"--- [{tag}] Stage 1: Calculating Global Weights ---")
        pts_list, etas_list, labels_list = [], [], []
        for r_path in all_root_files:
            with uproot.open(r_path) as f:
                tree = f["FlatSubstructureJetTree"]
                pts_list.append(tree["fjet_pt"].array(library="np"))
                etas_list.append(tree["fjet_eta"].array(library="np"))
                labels_list.append(tree["labels"].array(library="np"))
        pts, etas, labels = np.concatenate(pts_list), np.concatenate(etas_list), np.concatenate(labels_list)

        do_2d = config.get("flatten_pt", False) and config.get("flatten_eta", False)
        if do_2d:
            all_weights_sig = assign_2d_flat_weights_kde(etas[labels==1], pts[labels==1])
            all_weights_bkg = assign_2d_flat_weights_kde(etas[labels==0], pts[labels==0])
        else:
            all_weights_sig = assign_flat_weights(pts[labels==1], n_bins=config["n_bins_pt"])
            all_weights_bkg = assign_flat_weights(pts[labels==0], n_bins=config["n_bins_pt"])
        sig_ptr, bkg_ptr = 0, 0

    # Stage 2: Transform and Statistics Print
    print(f"\n--- [{tag}] Stage 2: Applying Transforms ---")
    # Print Next Operation 
    mode_str = "2D Flatten (pT & eta)" if (do_flatten and config.get("flatten_eta")) else "1D Flatten (pT only)" if do_flatten else "No Flatten (Test set)"
    print(f"[Action] Standardization: Enabled using global train stats.")
    print(f"[Action] Reweighting: {mode_str}")

    for idx, g_path in enumerate(all_graph_files):
        dataset = torch.load(g_path, weights_only=False)
        
        for g in dataset:
            # 1. Std
            g.x = torch.from_numpy((g.x.numpy() - mean_x) / std_x).float()
            epsilon = 1e-8 # To prevent division by zero in case of zero std
            g.Ntrk = torch.tensor((float(g.Ntrk) - mean_ntrk) / (std_ntrk + epsilon)).float()
            # 2. Flatten
            if do_flatten:
                if g.y == 1: g.weights = float(all_weights_sig[sig_ptr]); sig_ptr += 1
                else: g.weights = float(all_weights_bkg[bkg_ptr]); bkg_ptr += 1
            else: g.weights = 1.0

        # --- Statistical Snapshot per Batch ---
        all_x = torch.cat([data.x for data in dataset], dim=0).numpy()
        all_w = np.array([data.weights for data in dataset])
        print(f"  Batch {idx} | Mean(x): {np.mean(all_x):.4f} (exp~0) | Std(x): {np.std(all_x):.4f} (exp~1) | WeightAvg: {np.mean(all_w):.4f}")

        torch.save(dataset, os.path.join(out_dir, f"processed_{os.path.basename(g_path)}"))
        del dataset

    if tag == "train" and do_flatten:
        _save_plots(pts, etas, labels, all_weights_sig, all_weights_bkg, config, out_dir)

# ============================================================
# 3. Main Program
# ============================================================
def preprocess_SRJ_CPU(config):
    stats = json.load(open(config["data"]["meanstd_json"]))
    mean_x, std_x = np.array(stats["mean_x"], dtype=np.float32), np.array(stats["std_x"], dtype=np.float32)
    mean_ntrk, std_ntrk = float(stats["mean_ntrk"]), float(stats["std_ntrk"])

    base_out_dir = config["data"]["out_dir"]
    train_out = os.path.join(base_out_dir, "train")
    process_one_split(config["data"]["train_graphs"], config, mean_x, std_x, mean_ntrk, std_ntrk, True, "train", train_out)

    if "test_graphs" in config["data"]:
        test_out = os.path.join(base_out_dir, "test")
        process_one_split(config["data"]["test_graphs"], config, mean_x, std_x, mean_ntrk, std_ntrk, False, "test", test_out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    preprocess_SRJ_CPU(load_yaml(args.config))