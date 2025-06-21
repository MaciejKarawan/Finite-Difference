#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heatmap_dividend_amount.py

Generates an interactive heatmap focusing on DIVIDEND AMOUNT.
Option Price vs. Asset Price vs. Time to Expiration.
Uses FDB_with_memory and multiprocessing.

Slider:
1. Dividend Amount (fixed value)
"""

import numpy as np
import plotly.graph_objects as go
import itertools
import time
import os
import sys
import multiprocessing as mp
from functools import partial

# --- Configuration for Importing FDB_with_memory ---
FDB_MODULE_PATH = "/home/ubuntu/upload"
if FDB_MODULE_PATH not in sys.path:
    sys.path.append(FDB_MODULE_PATH)

try:
    from FD_KO_dividend import FDB_with_memory
    # Add dummy functions if they are not present in the imported module to avoid errors
    # This is a workaround for environments where the module might be incomplete
    if not hasattr(sys.modules.get("FD_KO_dividend"), "option_barrier_price"):
        def option_barrier_price(*args, **kwargs):
            print("Dummy option_barrier_price called")
            return None # Or some default value
        if "FD_KO_dividend" in sys.modules:
            sys.modules["FD_KO_dividend"].option_barrier_price = option_barrier_price

    if not hasattr(sys.modules.get("FD_KO_dividend"), "interpolate_option_price"):
        def interpolate_option_price(*args, **kwargs):
            print("Dummy interpolate_option_price called")
            return None # Or some default value
        if "FD_KO_dividend" in sys.modules:
            sys.modules["FD_KO_dividend"].interpolate_option_price = interpolate_option_price

except ImportError as e:
    print(f"Error importing FDB_with_memory from FD_KO_dividend.py: {e}")
    print(f"Please ensure FD_KO_dividend.py and its dependencies (interpolation.py, pricing.py) are in {FDB_MODULE_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# --- Main Configuration Section --- 
def get_config():
    config = {
        # Fixed parameters for FDB_with_memory simulation
        "s0": 2194.05,
        "r": np.log(1 + 0.05),
        "expiration": 1.0,
        "NAS": 100,
        "option_type": "a_put",

        # Fixed parameters from original dividend script
        "fixed_strike": 2200.0,
        "fixed_barrier": 1900.0, # From user's uploaded heatmap_dividend.py
        "fixed_vol_mid": 0.25,
        "fixed_vol_length": 0.1,
        "fixed_dividend_present": True, 
        "fixed_dividend_time_val": 0.5, # Dividend time is now fixed

        # Slider for THIS script: Dividend Amount
        "div_amount_slider": [500, 100, 50],  # Min, Max, NumSteps (from user's file)
        
        "output_html_file": "heatmap_dividend_amount.html",
        "num_processes": max(1, int(mp.cpu_count() * 0.8))
    }
    return config

# --- Frame Name Generation (specific to this script's slider) ---
def get_frame_name(div_amount_val):
    return f"Da{div_amount_val:.2f}"

# --- Worker function for multiprocessing ---
def process_grid_combination(div_amount_val, cfg):
    # div_amount_val is the single dynamic parameter
    div_time = cfg["fixed_dividend_time_val"]
    
    min_vol = cfg["fixed_vol_mid"] - cfg["fixed_vol_length"] / 2
    max_vol = cfg["fixed_vol_mid"] + cfg["fixed_vol_length"] / 2
    if min_vol < 0.001: min_vol = 0.001
    if max_vol <= min_vol: max_vol = min_vol + 0.001
    current_vol_range = [min_vol, max_vol]

    try:
        _, _, init_data, grid_data = FDB_with_memory(
            NAS=cfg["NAS"],
            s0=cfg["s0"],
            vol=current_vol_range,
            r=cfg["r"],
            option_type=cfg["option_type"],
            strike=cfg["fixed_strike"],
            expiration=cfg["expiration"],
            barrier=cfg["fixed_barrier"],
            is_dividend=cfg["fixed_dividend_present"],
            dividend_time=div_time, # Fixed value
            dividend_value=div_amount_val # Dynamic based on slider
        )
        result_data = {
            "S_vector": init_data["S"],
            "time_vector": init_data["time_grid"],
            "price_grid": grid_data["Grid"]
        }
        return div_amount_val, result_data
    except Exception as e:
        print(f"  Error in worker for params Da={div_amount_val:.2f}: {e}")
        return div_amount_val, None

# --- Data Generation (with multiprocessing) --- 
def precompute_all_grids(cfg):
    div_amount_values = np.linspace(cfg["div_amount_slider"][0], cfg["div_amount_slider"][1], cfg["div_amount_slider"][2])

    total_combinations = len(div_amount_values)
    print(f"Total DIVIDEND AMOUNT parameter values to compute: {total_combinations}")
    print(f"Using {cfg['num_processes']} processes.")
    
    computed_grids = {}
    start_total_time = time.time()
    worker_func_partial = partial(process_grid_combination, cfg=cfg)

    completed_count = 0
    with mp.Pool(processes=cfg['num_processes']) as pool:
        for div_amount_val, result_data in pool.imap_unordered(worker_func_partial, div_amount_values):
            computed_grids[div_amount_val] = result_data
            completed_count += 1
            print(f"Computed grid {completed_count}/{total_combinations} for Da={div_amount_val:.2f}...")

    end_total_time = time.time()
    print(f"Finished precomputing DIVIDEND AMOUNT grids in {end_total_time - start_total_time:.2f} seconds.")
    return computed_grids, div_amount_values

# --- Plotting --- 
def create_interactive_heatmap(cfg, all_grids_data, div_amt_vals):
    initial_div_amt = div_amt_vals[0]
    initial_grid_info = all_grids_data.get(initial_div_amt)

    if initial_grid_info is None:
        valid_key = next((k for k, v in all_grids_data.items() if v is not None), None)
        if valid_key:
            initial_div_amt = valid_key
            initial_grid_info = all_grids_data[valid_key]
        else:
            print("Error: No valid precomputed grids for DIVIDEND AMOUNT. Cannot create heatmap.")
            return

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=initial_grid_info["time_vector"],
        y=initial_grid_info["S_vector"][::-1],
        z=initial_grid_info["price_grid"][::-1, :],
        colorscale="Viridis",
        colorbar={"title": "Option Price"},
        name=get_frame_name(initial_div_amt)
    ))

    frames = []
    for div_amount_val, grid_info in all_grids_data.items():
        if grid_info:
            frame_name = get_frame_name(div_amount_val)
            frames.append(go.Frame(
                data=[go.Heatmap(
                    x=grid_info["time_vector"],
                    y=grid_info["S_vector"][::-1],
                    z=grid_info["price_grid"][::-1, :],
                    colorscale="Viridis", colorbar={"title": "Option Price"}
                )],
                name=frame_name
            ))
    if frames:
        fig.frames = frames

    sliders_definitions = []
    # Slider 1: Dividend Amount
    steps_da = []
    for val in div_amt_vals:
        target_frame_name = get_frame_name(val)
        steps_da.append({
            "label": f"{val:.2f}", "method": "animate",
            "args": [[target_frame_name], {"mode": "immediate", "frame": {"duration": 50, "redraw": True}}]
        })
    sliders_definitions.append({
        "active": 0, "currentvalue": {"prefix": "Div Amt: ", "visible": True, "xanchor": "right"},
        "pad": {"t": 50, "b":10}, "steps": steps_da, "len": 0.9, "x": 0.05, "y": -0.15 # Adjusted y and pad
    })

    updatemenus = [
        dict(type="buttons", direction="left", x=0.5, xanchor="center", y=-0.05, yanchor="top", showactive=False, # Adjusted y
            buttons=[
                dict(label="Play All Frames", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ])]

    fig.update_layout(
        title=f"Interactive Heatmap (Dividend Amount): {cfg['option_type']}<br><sup>K={cfg['fixed_strike']}, B={cfg['fixed_barrier']}, VolMid={cfg['fixed_vol_mid']:.2f}, VolLen={cfg['fixed_vol_length']:.2f}, DivTime={cfg['fixed_dividend_time_val']:.2f}</sup>",
        xaxis_title="Time to Expiration (Years)", yaxis_title="Asset Price (S)",
        sliders=sliders_definitions, updatemenus=updatemenus,
        height=700, # Adjusted height for single slider layout
        width=800  # From user's file
    )
    
    try:
        fig.write_html(cfg["output_html_file"])
        print(f"Interactive DIVIDEND AMOUNT heatmap saved to {cfg['output_html_file']}")
    except Exception as e:
        print(f"Error saving DIVIDEND AMOUNT heatmap: {e}")

if __name__ == "__main__":
    mp.freeze_support()
    config = get_config()
    print("Starting DIVIDEND AMOUNT heatmap generation...")
    all_grids, div_amt_v = precompute_all_grids(config)
    if not all_grids or all(grid_data is None for grid_data in all_grids.values()):
        print("No DIVIDEND AMOUNT grid data computed. Exiting.")
    else:
        print("--- Creating DIVIDEND AMOUNT Interactive Heatmap --- ")
        create_interactive_heatmap(config, all_grids, div_amt_v)
    print("--- DIVIDEND AMOUNT Script Finished ---")

