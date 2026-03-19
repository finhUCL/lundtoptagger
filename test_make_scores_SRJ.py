import argparse
import os
import glob
import time
import gc

import uproot
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import load_yaml, get_scores
from tools.utils_config import recursive_update, parse_dot_args

print("Libraries loaded!")

def build_model_by_name(name):
    if name == "LundNet": return LundNet()
    elif name == "GATNet": return GATNet()
    elif name == "GINNet": return GINNet()
    elif name == "EdgeGinNet": return EdgeGinNet()
    elif name == "PNANet": return PNANet()
    elif name == "LundNet_plus_GN2X": return LundNet_plus_GN2X()
    elif name == "LundNet_plus_GN3X": return LundNet_plus_GN3X()
    elif name == "QLundNet": return QLundNet()
    else: raise ValueError(f"Unknown model type {name}")

def merge_root_files(file_list, output_name, tree_name):
    """
    Merge ROOT files in a pure-Python environment with uproot.  
    """
    if not file_list:
        return
    
    print(f"  -> Collecting all data for merge: {os.path.basename(output_name)}")
    
    # 1. Collect all shard data into a single list
    all_data = []
    for i, f_path in enumerate(file_list):
        with uproot.open(f_path) as f_in:
            data = f_in[tree_name].arrays()
            all_data.append(data)
            print(f"     Read shard {i+1}/{len(file_list)}")

    # 2. Use `awkward.concatenate` to merge these arrays
    import awkward as ak
    print("  -> Concatenating arrays...")
    combined_data = ak.concatenate(all_data)

    # 3. Write the final file in one go
    print(f"  -> Writing to {os.path.basename(output_name)}...")
    with uproot.recreate(output_name) as f_out:
        f_out[tree_name] = combined_data
    
    print("  -> Merge complete.")

def main():
    parser = argparse.ArgumentParser(description='Run SRJ scoring with Automatic Shard Merging')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    parser.add_argument('--override', nargs='*', default=[])
    args = parser.parse_args()

    config = load_yaml(args.config)
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    kT_selection = config['data']['kT_cut']
    placeholder = dict(
        sample=config['data']['sample'],
        kT_cut=kT_selection
    )

    # 1. Parse file
    paths_graphs = config['data']['paths_to_test_file_graphs']
    if isinstance(paths_graphs, str): paths_graphs = [paths_graphs]
    files_graphs = []
    for p in paths_graphs:
        files_graphs.extend(glob.glob(p.format(**placeholder)))
    files_graphs.sort()

    paths_root = config['data']['paths_to_test_file_root']
    if isinstance(paths_root, str): paths_root = [paths_root]
    files_root = []
    for p in paths_root:
        files_root.extend(glob.glob(p.format(**placeholder)))
    files_root.sort()

    if len(files_graphs) != len(files_root):
        raise ValueError(f"Mismatch: {len(files_graphs)} graphs vs {len(files_root)} roots!")

    # 2. Output Settings
    outdir = config['data']['path_to_outdir'].format(**placeholder)
    os.makedirs(outdir, exist_ok=True)
    output_suffix = config['data']['output_suffix'].format(**placeholder)
    
    tree_name = "FlatSubstructureJetTree"
    batch_size = config['test']['batch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    t_global = time.time()
    generated_shards = []

    # 3. Main Evaluation Loop
    for file_number, (fg, fr) in enumerate(zip(files_graphs, files_root), start=1):
        print(f"\n[{file_number}/{len(files_graphs)}] Processing Shard: {os.path.basename(fg)}")

        dataset = torch.load(fg, weights_only=False)
        n_jets = len(dataset)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        multi_scores = {}

        for m in config["test"]["models_to_run"]:
            tag, arch, ckpt = m["tag"], m["arch"], m["ckpt"]
            print(f"  -> Scoring model: {tag} ({arch})")
            
            model = build_model_by_name(arch)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device)
            model.eval()

            y_pred = get_scores(test_loader, model, device)
            multi_scores[tag] = np.array(y_pred[:, 0])

            del model, y_pred
            torch.cuda.empty_cache()

        shard_out_name = os.path.join(outdir, os.path.basename(fr).replace(".root", f"{output_suffix}.root"))
        
        with uproot.open(fr) as f_in:
            arrays = f_in[tree_name].arrays()

        if len(arrays) != n_jets:
            print(f"CRITICAL ERROR: Entry mismatch in {os.path.basename(fg)}!")
            continue

        for tag, scores in multi_scores.items():
            branch_name = config["test"]["scores_branch_name"].format(tag=tag)
            arrays[branch_name] = scores

        print(f"  Saving shard to: {os.path.basename(shard_out_name)}")
        with uproot.recreate(shard_out_name) as f_out:
            f_out[tree_name] = arrays
        
        generated_shards.append(shard_out_name)
        del dataset, arrays
        gc.collect()

    # ---------------------------------------------------------
    # 4. Merge shards into a single ROOT file (pure-Python merge with uproot)
    # ---------------------------------------------------------
    if generated_shards:
        final_outfile = os.path.join(outdir, f"Final_Scores_{config['data']['sample']}_kT{kT_selection}.root")
        print(f"\n=== Merging {len(generated_shards)} shards into single file (Pure Python) ===")
        
        try:
            merge_root_files(generated_shards, final_outfile, tree_name)
            print("Successfully created combined ROOT file.")
            
            # --- Delete temporary shards only if the merge succeeds ---
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