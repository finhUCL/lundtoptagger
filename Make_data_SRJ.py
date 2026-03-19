import argparse
import csv
from datetime import datetime
import os
import json
import glob
import time
import gc

import uproot
import awkward as ak
import numpy as np
import torch

from tools.GNN_model_weight.utils_newdata import (
    load_yaml,
    srj_create_train_dataset_fulld_new_Ntrk_pt_file
)
from tools.utils_config import recursive_update, parse_dot_args

print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for classifier input (SRJ batch version)")
    add_arg = parser.parse_args
    parser.add_argument("config", help="job configuration file")
    parser.add_argument('--override', nargs='*', default=[], help='Overrides in the form key.subkey=value')

    args = parser.parse_args()
    config = load_yaml(args.config)
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    # Batch saving settings
    max_jets_per_batch = config.get("max_jets_per_batch", 10000)

    # Signal config
    config_signal = load_yaml(config["signal_config_file"])
    signal = config["signal"]
    signals = [s for s in config_signal.keys() if s != "bkg_histos"] if signal == "all" else [signal]

    raw_paths = config["path_to_rootfiles"]
    path_list = [raw_paths] if isinstance(raw_paths, str) else raw_paths
    
    files = []
    for pat in path_list:
        files.extend(glob.glob(pat))
    
    if config.get("n_files") is not None:
        files = files[:config["n_files"]]

    n_files = len(files)
    print(f"Processing {n_files} files")
    if n_files == 0:
        print("Error: No files found. Please check 'path_to_rootfiles' in your config.")
        return

    event_fractions = []
    for frac, n_chunks in config["event_fractions"].items():
        event_fractions.extend([frac] * n_chunks)
    
    event_fraction_indices = [config["event_fraction_idx"]] if config["event_fraction_idx"] is not None else list(range(len(event_fractions)))

    # Mapping definition
    jet_property_names = {
        "fjet_m": "SRJ_mass", "fjet_pt": "SRJ_pt", "fjet_eta": "SRJ_eta",
        "fjet_phi": "SRJ_phi", "fjet_truth_label": "SRJ_partonTruthLabel",
        "fjet_Nconst": "SRJ_Nconst", "fjet_Nconst_Charged": "SRJ_Nconst_Charged",
        # can comment out if using old dataset without transformer scores
        "parT_score": "SRJ_QGScore",
    }
    additional_output_vars = ["EventInfo_mcEventWeight", "EventInfo_mcChannelNumber"]

    for frac_idx in event_fraction_indices:
        event_fraction = event_fractions[frac_idx]
        percent_str = f"{event_fraction * 100:.2f}".rstrip('0').rstrip('.') + "percent"
        folder_name = f"part{frac_idx}_{percent_str}"
        
        # Base output path handling
        # 1. First replace {id} in the base path with the specific value (e.g., "dijet")
        base_path = config["out_dir"].format(id=config["id"]) 

        # 2. Then append the desired partX_Ypercent folder
        # This prevents creating redundant {frac} directory levels
        final_out_dir = os.path.join(base_path, f"part{frac_idx}_{percent_str}")

        os.makedirs(final_out_dir, exist_ok=True)
        
        print(f"\n--- Processing Part {frac_idx} (Fraction: {event_fraction}) ---")
        print(f"--- Saving to: {final_out_dir} ---")

        dataset = []
        primary_Lund_only_one_arr = []
        batch_idx = 0

        # Welford statistics (global)
        feat_dim, slice_count, slice_mean, slice_M2 = 3, 0, np.zeros(3), np.zeros(3)
        slice_count_ntrk, slice_mean_ntrk, slice_M2_ntrk = 0, 0.0, 0.0

        def welford_update(mean, M2, count, x):
            count += 1
            delta = x - mean
            mean += delta / count
            M2 += delta * (x - mean)
            return mean, M2, count

        # Initialize tree dictionary
        out_tree_dict = {name: ak.Array([]) for name in [*jet_property_names.keys(), *additional_output_vars]}

        def save_batch(data_list, tree_dict, b_idx):
            if len(data_list) == 0: return
            
            batch_placeholders = dict(
                id=config["id"], kT_cut=config["kT_cut"],
                include_pt="_with_pt" if config["include_pt"] else "",
                frac=f"_batch{b_idx}"
            )

            # 1. Save Graph
            g_filename = config["out_file_name_graphs"].format(**batch_placeholders)
            g_path = os.path.join(final_out_dir, g_filename)
            torch.save(data_list, g_path)

            # 2. Save ROOT (sync physical variables)
            tree_dict["labels"] = ak.Array([g.y for g in data_list])
            r_filename = config["out_file_name_root"].format(**batch_placeholders)
            r_path = os.path.join(final_out_dir, r_filename)
            with uproot.recreate(r_path) as f:
                f["FlatSubstructureJetTree"] = tree_dict
            
            print(f">>> Batch {b_idx} saved: {len(data_list)} jets.")

        # Loop over files
        for file_num, file in enumerate(files, start=1):
            with uproot.open(file) as infile:
                tree = infile["AnalysisTree"]
                dsids = tree["dsid"].array(library="np")
                if dsids[0] in set.intersection(*[set(config_signal[s]["skip_dsids"]) for s in signals]): continue

                total_e = tree.num_entries
                start = int(total_e * sum(event_fractions[:frac_idx]))
                stop = int(total_e * (sum(event_fractions[:frac_idx]) + event_fraction))
                if start >= stop: continue

                # Load Jet properties
                jet_props = {}
                for out_n, in_n in jet_property_names.items():
                    jet_props[in_n] = ak.flatten(tree[in_n].array(entry_start=start, entry_stop=stop, library="ak"))
                
                # Load Lund variables
                lund_maps = {"SRJ_jetLundZ":"jetLundZ", "SRJ_jetLundKt":"jetLundKt", "SRJ_jetLundDeltaR":"jetLundDeltaR", "SRJ_jetLundIDParent1":"jetLundIDParent1", "SRJ_jetLundIDParent2":"jetLundIDParent2"}
                for srj, inn in lund_maps.items():
                    jet_props[inn] = ak.flatten(tree[srj].array(entry_start=start, entry_stop=stop, library="ak"))

                # Weights and DSID expansion
                n_jets = ak.num(tree["SRJ_partonTruthLabel"].array(entry_start=start, entry_stop=stop))
                jet_props["EventInfo_mcEventWeight"] = np.repeat(tree["mcEventWeight"].array(entry_start=start, entry_stop=stop, library="np"), n_jets)
                jet_props["EventInfo_mcChannelNumber"] = np.repeat(dsids[start:stop], n_jets)

                # Selection criteria
                pt_min = min(config_signal[s]["pt_range"][0] for s in signals)
                pt_max = max(config_signal[s]["pt_range"][1] for s in signals)
                e_min = min(config_signal[s].get("eta_min", 0.0) for s in signals)
                e_max = max(config_signal[s]["eta_max"] for s in signals)
                min_sp = min(config_signal[s]["min_splits"] for s in signals)

                # Welford statistics update (only for jets passing basic selection)
                mask = (jet_props["SRJ_pt"] > pt_min) & (jet_props["SRJ_pt"] < pt_max) & \
                       (np.abs(jet_props["SRJ_eta"]) > e_min) & (np.abs(jet_props["SRJ_eta"]) < e_max) & \
                       (ak.num(jet_props["jetLundZ"]) >= min_sp)

                filtered_ntrk = ak.to_numpy(jet_props["SRJ_Nconst_Charged"][mask]).astype(float)
                for vv in filtered_ntrk: 
                    slice_mean_ntrk, slice_M2_ntrk, slice_count_ntrk = welford_update(slice_mean_ntrk, slice_M2_ntrk, slice_count_ntrk, vv)

                raw_nodes = np.vstack([
                    np.log(1.0 / (ak.to_numpy(ak.flatten(jet_props["jetLundDeltaR"][mask])) + 1e-4)),
                    np.log(1.0 / (ak.to_numpy(ak.flatten(jet_props["jetLundZ"][mask])) + 1e-4)),
                    np.log(ak.to_numpy(ak.flatten(jet_props["jetLundKt"][mask])) + 1e-4)
                ]).T
                for v in raw_nodes: 
                    slice_mean, slice_M2, slice_count = welford_update(slice_mean, slice_M2, slice_count, v)

                # Construct Graph list
                passed_selection = []
                dataset = srj_create_train_dataset_fulld_new_Ntrk_pt_file(
                    dataset, jet_props["jetLundZ"], jet_props["jetLundKt"], jet_props["jetLundDeltaR"],
                    jet_props["jetLundIDParent1"], jet_props["jetLundIDParent2"], jet_props["SRJ_partonTruthLabel"],
                    jet_props["EventInfo_mcChannelNumber"], jet_props["SRJ_Nconst_Charged"], jet_props["SRJ_pt"],
                    jet_props["SRJ_mass"], jet_props["SRJ_eta"], kT_selection=config["kT_cut"],
                    primary_Lund_only_one_arr=primary_Lund_only_one_arr, passed_selection=passed_selection,
                    signal_jet_truth_labels=set().union(*[config_signal[s]["signal_jet_truth_labels"] for s in signals]),
                    signal_dsids=set().union(*[config_signal[s]["dsids"] for s in signals]),
                    pt_range=(pt_min, pt_max), mass_range=(0, 1e6), eta_min=e_min, eta_max=e_max, min_splits=min_sp, include_pt=config["include_pt"]
                )

                # Fill ROOT dictionary (only save passed events)
                for out_n, in_n in jet_property_names.items():
                    out_tree_dict[out_n] = ak.concatenate([out_tree_dict[out_n], jet_props[in_n][passed_selection]])
                for var in ["EventInfo_mcEventWeight", "EventInfo_mcChannelNumber"]:
                    out_tree_dict[var] = ak.concatenate([out_tree_dict[var], jet_props[var][passed_selection]])

                # Check for batch saving
                if len(dataset) >= max_jets_per_batch:
                    save_batch(dataset, out_tree_dict, batch_idx)
                    dataset, batch_idx = [], batch_idx + 1
                    out_tree_dict = {name: ak.Array([]) for name in out_tree_dict}
                    gc.collect()

        # Process remaining data
        if len(dataset) > 0:
            save_batch(dataset, out_tree_dict, batch_idx)

        # Save global statistics for this Part
        stats = {
            "mean_x": slice_mean.tolist(), 
            "std_x": np.sqrt(slice_M2 / max(slice_count - 1, 1)).tolist(),
            "mean_ntrk": float(slice_mean_ntrk), 
            "std_ntrk": float(np.sqrt(slice_M2_ntrk / max(slice_count_ntrk - 1, 1))),
            "nodes": int(slice_count), "jets": int(slice_count_ntrk)
        }
        json_path = os.path.join(final_out_dir, f"meanstd_part{frac_idx}.json")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Part {frac_idx} complete. Stats saved to {json_path}")

if __name__ == "__main__":
    main()