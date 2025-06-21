#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heatmap_volatility.py

Generates an interactive heatmap focusing on VOLATILITY parameters.
Option Price vs. Asset Price vs. Time to Expiration.
Uses FDB_with_memory and multiprocessing.

Sliders:
1. Midpoint of Uncertain Volatility
2. Length of Uncertain Volatility Interval
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
    if 'option_barrier_price' not in sys.modules.get('FD_KO_dividend', {}).__dict__:
        def option_barrier_price(*args, **kwargs):
            return None
        if 'FD_KO_dividend' in sys.modules:
            sys.modules['FD_KO_dividend'].option_barrier_price = option_barrier_price
    if 'interpolate_option_price' not in sys.modules.get('FD_KO_dividend', {}).__dict__:
        def interpolate_option_price(*args, **kwargs):
            return None
        if 'FD_KO_dividend' in sys.modules:
            sys.modules['FD_KO_dividend'].interpolate_option_price = interpolate_option_price
except ImportError as e:
    print(f"Error importing FDB_with_memory from FD_KO_dividend.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# --- Main Configuration Section --- 
def get_config():
    config = {
        # Fixed parameters for FDB_with_memory simulation
        "s0": 2000.0,
        "r": np.log(1 + 0.05),
        "expiration": 1.0,
        "NAS": 100, # Reduced NAS for faster individual script testing initially
        "option_type": "e_put", # As requested

        # Fixed parameters (previously sliders, now fixed for this script)
        "fixed_strike": 2200.0,
        "fixed_barrier": 1900.0, # For a_call, barrier > s0 and > strike typically
        "fixed_dividend_present": True,
        "fixed_dividend_amount": 500.0, # Fixed amount
        "fixed_dividend_time": 0.5,    # Fraction of expiration

        # Sliders for THIS script: Volatility Midpoint and Length
        "vol_mid_slider": [0.1, 0.4, 10],  # Min, Max, NumSteps
        "vol_length_slider": [0.05, 0.3, 10], # Min, Max, NumSteps
        
        "output_html_file": "heatmap_volatility.html",
        "num_processes": max(1, int(mp.cpu_count() * 0.8))
    }
    return config

# --- Frame Name Generation (specific to this script's sliders) ---
def get_frame_name(params_tuple):
    vm, vl = params_tuple
    return f"Vm{vm:.3f}_Vl{vl:.3f}"

# --- Worker function for multiprocessing ---
def process_grid_combination(params_tuple, cfg):
    vol_mid, vol_length = params_tuple
    # frame_name = get_frame_name(params_tuple) # For debugging worker

    min_vol = vol_mid - vol_length / 2
    max_vol = vol_mid + vol_length / 2
    if min_vol < 0.001: min_vol = 0.001
    if max_vol <= min_vol: max_vol = min_vol + 0.001 # Ensure max_vol > min_vol
    current_vol_range = [min_vol, max_vol]

    try:
        _, _, init_data, grid_data = FDB_with_memory(
            NAS=cfg["NAS"],
            s0=cfg["s0"],
            vol=current_vol_range, # Dynamic based on sliders
            r=cfg["r"],
            option_type=cfg["option_type"],
            strike=cfg["fixed_strike"], # Fixed
            expiration=cfg["expiration"],
            barrier=cfg["fixed_barrier"], # Fixed
            is_dividend=cfg["fixed_dividend_present"], # Fixed
            dividend_time=cfg["fixed_dividend_time"], # Fixed
            dividend_value=cfg["fixed_dividend_amount"] # Fixed
        )
        result_data = {
            "S_vector": init_data["S"],
            "time_vector": init_data["time_grid"],
            "price_grid": grid_data["Grid"]
        }
        return params_tuple, result_data
    except Exception as e:
        print(f"  Error in worker for params Vm={vol_mid:.3f}, Vl={vol_length:.3f}: {e}")
        return params_tuple, None

# --- Data Generation (with multiprocessing) --- 
def precompute_all_grids(cfg):
    vol_mid_values = np.linspace(cfg["vol_mid_slider"][0], cfg["vol_mid_slider"][1], cfg["vol_mid_slider"][2])
    vol_length_values = np.linspace(cfg["vol_length_slider"][0], cfg["vol_length_slider"][1], cfg["vol_length_slider"][2])

    param_combinations = list(itertools.product(vol_mid_values, vol_length_values))

    total_combinations = len(param_combinations)
    print(f"Total VOLATILITY parameter combinations to compute: {total_combinations}")
    print(f"Using {cfg['num_processes']} processes.")
    
    computed_grids = {}
    start_total_time = time.time()
    worker_func_partial = partial(process_grid_combination, cfg=cfg)

    completed_count = 0
    with mp.Pool(processes=cfg['num_processes']) as pool:
        for params_tuple, result_data in pool.imap_unordered(worker_func_partial, param_combinations):
            computed_grids[params_tuple] = result_data
            completed_count += 1
            print(f"Computed grid {completed_count}/{total_combinations} for Vm={params_tuple[0]:.3f}, Vl={params_tuple[1]:.3f}...")

    end_total_time = time.time()
    print(f"Finished precomputing VOLATILITY grids in {end_total_time - start_total_time:.2f} seconds.")
    # Return all slider value lists for consistency, though only two are used here for combinations
    return computed_grids, vol_mid_values, vol_length_values

# --- Plotting --- 
def create_interactive_heatmap(cfg, all_grids_data, vol_mid_vals, vol_len_vals):
    initial_params_tuple = (vol_mid_vals[0], vol_len_vals[0])
    initial_grid_info = all_grids_data.get(initial_params_tuple)

    if initial_grid_info is None:
        valid_key = next((k for k, v in all_grids_data.items() if v is not None), None)
        if valid_key:
            initial_params_tuple = valid_key
            initial_grid_info = all_grids_data[valid_key]
        else:
            print("Error: No valid precomputed grids for VOLATILITY. Cannot create heatmap.")
            return

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=initial_grid_info["time_vector"],
        y=initial_grid_info["S_vector"][::-1],
        z=initial_grid_info["price_grid"][::-1, :],
        colorscale="Viridis",
        colorbar={"title": "Option Price"},
        name=get_frame_name(initial_params_tuple)
    ))

    frames = []
    for params_tuple, grid_info in all_grids_data.items():
        if grid_info:
            frame_name = get_frame_name(params_tuple)
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
    # Slider 1: Volatility Midpoint
    steps_vm = []
    for val in vol_mid_vals:
        # Frame name depends on BOTH sliders' initial values, except for the one being changed
        target_frame_name = get_frame_name((val, vol_len_vals[0]))
        steps_vm.append({
            "label": f"{val:.3f}", "method": "animate",
            "args": [[target_frame_name], {"mode": "immediate", "frame": {"duration": 100, "redraw": True}}]
        })
    sliders_definitions.append({
        "active": 0, "currentvalue": {"prefix": "Vol Mid: ", "visible": True},
        "pad": {"t": 20}, "steps": steps_vm, "len": 0.9, "x": 0.05, "y": 0
    })

    # Slider 2: Volatility Length
    steps_vl = []
    for val in vol_len_vals:
        target_frame_name = get_frame_name((vol_mid_vals[0], val))
        steps_vl.append({
            "label": f"{val:.3f}", "method": "animate",
            "args": [[target_frame_name], {"mode": "immediate", "frame": {"duration": 100, "redraw": True}}]
        })
    sliders_definitions.append({
        "active": 0, "currentvalue": {"prefix": "Vol Length: ", "visible": True},
        "pad": {"t": 80}, "steps": steps_vl, "len": 0.9, "x": 0.05, "y": 0
    })
    
    # Simplified slider interaction: Each slider step directly targets a frame.
    # This assumes frames are named based on the combination of (slider1_val, slider2_val_initial) or (slider1_val_initial, slider2_val)
    # For full interactivity where sliders are independent, a more complex frame naming or JavaScript is needed.
    # The current approach is inspired by the user's first script, where sliders effectively select a pre-rendered frame.
    # To make sliders truly independent, each step would need to find a frame matching (current_slider1_val, this_slider2_step_val)
    # This would require JS or a more complex `args` structure if all frames for all combinations are generated.
    # The current `get_frame_name` and `param_combinations` in `precompute_all_grids` *do* generate all combinations.
    # So, the slider steps *should* target these fully combined frame names.

    # Corrected slider step generation for full interactivity with 2 sliders:
    sliders_definitions = []
    # Slider 1: Vol Midpoint
    steps_vm_corr = []
    active_vol_len_idx = 0 # Assuming the other slider is at its initial position for this step's target frame name
    for vm_val in vol_mid_vals:
        target_frame_name = get_frame_name((vm_val, vol_len_vals[active_vol_len_idx]))
        steps_vm_corr.append({
            "label": f"{vm_val:.3f}", "method": "animate",
            "args": [[target_frame_name], {"mode": "immediate", "frame": {"duration": 50, "redraw": True}}]
        })
    sliders_definitions.append({
        "active": 0, "currentvalue": {"prefix": "Vol Mid: ", "visible": True, "xanchor": "right"},
        "pad": {"t": 20, "b": 10}, # Górne i dolne wypełnienie
        "steps": steps_vm_corr,
        "len": 0.8,  # Zwiększamy długość suwaka
        "x": 0.1,    # Wyrównujemy pozycję poziomą (10% od lewej)
        "y": -0.2    # Pozycja pionowa dla górnego suwaka (poniżej wykresu)
    })

    # Slider 2: Vol Length
    steps_vl_corr = []
    active_vol_mid_idx = 0 # Assuming the other slider is at its initial position
    for vl_val in vol_len_vals:
        target_frame_name = get_frame_name((vol_mid_vals[active_vol_mid_idx], vl_val))
        steps_vl_corr.append({
            "label": f"{vl_val:.3f}", "method": "animate",
            "args": [[target_frame_name], {"mode": "immediate", "frame": {"duration": 50, "redraw": True}}]
        })
    sliders_definitions.append({
        "active": 0, "currentvalue": {"prefix": "Vol Length: ", "visible": True, "xanchor": "right"},
        "pad": {"t": 60, "b": 10}, # Zwiększamy górne wypełnienie, aby zrobić miejsce pod pierwszym suwakiem
        "steps": steps_vl_corr,
        "len": 0.8,  # Taka sama długość jak górny suwak
        "x": 0.1,    # Taka sama pozycja pozioma jak górny suwak
        "y": -0.3    # Pozycja pionowa dla dolnego suwaka (niżej niż pierwszy)
    })
    # Note: The above slider logic is still simplified. True interdependent sliders in pure Plotly.go without Dash
    # typically require a more complex setup or a single "master" slider that iterates through combined states.
    # For two sliders, we can make them work if each step updates the frame assuming the *other* slider is at its *currently selected value*.
    # This is hard without JS. The most robust way with pure Plotly frames is to have each slider step point to a unique frame name
    # that represents the full combination of (vol_mid_value_of_this_step, vol_length_value_of_OTHER_slider_CURRENT_value).
    # The provided user's first script did this by naming frames with all parameters and sliders just picked one.
    # The current `frames` list contains all combinations. So, slider steps need to correctly reference these.

    # Final attempt at slider logic for 2 sliders, ensuring each step targets a unique, existing frame.
    # This assumes that when a slider is moved, the *other* slider's value is taken from its *current active step*.
    # This is not directly supported by the 'animate' method's args in a simple way for multiple independent sliders.
    # The simplest robust solution with frames is that each slider step defines the *full* set of parameters for the target frame.
    # The following uses the logic from the 5-slider version, adapted for 2 sliders.
    # It assumes changing one slider keeps the other at its *initial* (0-th index) setting for targeting a frame.
    # This will show the effect of one parameter while the other is fixed at its start value.
    # If user wants to see (slider1_pos3, slider2_pos2), they'd move slider1 to pos3, then slider2 to pos2, and the frame should update.
    # This requires the `animate` args to be constructed dynamically or all frames to be listed in each step. 
    # The user's original script had sliders that selected frames based on a string composed of all current param values.
    # We will stick to the simpler model where each slider step picks a frame based on its value and the *initial* values of other sliders.
    # This is less interactive than Dash but matches the original approach.

    updatemenus = [
        dict(type="buttons", direction="left", x=0.5, xanchor="center", y=-0.2, yanchor="top", showactive=False,
            buttons=[
                dict(label="Play All Frames", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ])]

    fig.update_layout(
        title=f"Interactive Heatmap (Volatility): {cfg['option_type']}<br><sup>K={cfg['fixed_strike']}, B={cfg['fixed_barrier']}, DivAmt={cfg['fixed_dividend_amount']}@T={cfg['fixed_dividend_time']}</sup>",
        xaxis_title="Time to Expiration (Years)", yaxis_title="Asset Price (S)",
        sliders=sliders_definitions, updatemenus=updatemenus,
        height=650 + len(sliders_definitions) * 40,
        width=800 + len(sliders_definitions) * 40
    )
    
    try:
        fig.write_html(cfg["output_html_file"])
        print(f"Interactive VOLATILITY heatmap saved to {cfg['output_html_file']}")
    except Exception as e:
        print(f"Error saving VOLATILITY heatmap: {e}")

if __name__ == "__main__":
    mp.freeze_support()
    config = get_config()
    print("Starting VOLATILITY heatmap generation...")
    all_grids, vol_mid_v, vol_len_v = precompute_all_grids(config)
    if not all_grids or all(grid_data is None for grid_data in all_grids.values()):
        print("No VOLATILITY grid data computed. Exiting.")
    else:
        print("--- Creating VOLATILITY Interactive Heatmap --- ")
        create_interactive_heatmap(config, all_grids, vol_mid_v, vol_len_v)
    print("--- VOLATILITY Script Finished ---")

