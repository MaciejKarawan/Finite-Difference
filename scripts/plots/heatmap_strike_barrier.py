from finite_difference.paths import generated_path
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heatmap_strike_barrier.py

Generates an interactive heatmap focusing on STRIKE and BARRIER parameters.
Option Price vs. Asset Price vs. Time to Expiration.
Uses FDB_with_memory and multiprocessing.

Sliders:
1. Strike Price
2. Barrier Value
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

try:
    from finite_difference.fd_ko_dividend import FDB_with_memory
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
        "NAS": 50, # Reduced NAS for faster individual script testing initially
        "option_type": "a_call", # As requested

        # Fixed parameters (previously sliders, now fixed for this script)
        "fixed_vol_mid": 0.25,
        "fixed_vol_length": 0.1,
        "fixed_dividend_present": True,
        "fixed_dividend_amount": 20.0,
        "fixed_dividend_time": 0.5,

        # Sliders for THIS script: Strike and Barrier
        "strike_slider": [1500.0, 3000.0, 30],  # Min, Max, NumSteps
        "barrier_slider": [1600.0, 3500.0, 30], # Min, Max, NumSteps
        
        "output_html_file": "heatmap_strike_barrier.html",
        "num_processes": max(1, int(mp.cpu_count() * 0.8))
    }
    return config

# --- Frame Name Generation (specific to this script's sliders) ---
def get_frame_name(params_tuple):
    k, b = params_tuple
    return f"K{k:.2f}_B{b:.2f}"

# --- Worker function for multiprocessing ---
def process_grid_combination(params_tuple, cfg):
    strike, barrier = params_tuple
    
    min_vol = cfg["fixed_vol_mid"] - cfg["fixed_vol_length"] / 2
    max_vol = cfg["fixed_vol_mid"] + cfg["fixed_vol_length"] / 2
    if min_vol < 0.001: min_vol = 0.001
    if max_vol <= min_vol: max_vol = min_vol + 0.001
    current_vol_range = [min_vol, max_vol]

    try:
        _, _, init_data, grid_data = FDB_with_memory(
            NAS=cfg["NAS"],
            s0=cfg["s0"],
            vol=current_vol_range, # Fixed
            r=cfg["r"],
            option_type=cfg["option_type"],
            strike=strike, # Dynamic based on slider
            expiration=cfg["expiration"],
            barrier=barrier, # Dynamic based on slider
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
        print(f"  Error in worker for params K={strike:.2f}, B={barrier:.2f}: {e}")
        return params_tuple, None

# --- Data Generation (with multiprocessing) --- 
def precompute_all_grids(cfg):
    strike_values = np.linspace(cfg["strike_slider"][0], cfg["strike_slider"][1], cfg["strike_slider"][2])
    barrier_values = np.linspace(cfg["barrier_slider"][0], cfg["barrier_slider"][1], cfg["barrier_slider"][2])

    param_combinations = list(itertools.product(strike_values, barrier_values))

    total_combinations = len(param_combinations)
    print(f"Total STRIKE/BARRIER parameter combinations to compute: {total_combinations}")
    print(f"Using {cfg['num_processes']} processes.")
    
    computed_grids = {}
    start_total_time = time.time()
    worker_func_partial = partial(process_grid_combination, cfg=cfg)

    completed_count = 0
    with mp.Pool(processes=cfg['num_processes']) as pool:
        for params_tuple, result_data in pool.imap_unordered(worker_func_partial, param_combinations):
            computed_grids[params_tuple] = result_data
            completed_count += 1
            print(f"Computed grid {completed_count}/{total_combinations} for K={params_tuple[0]:.2f}, B={params_tuple[1]:.2f}...")

    end_total_time = time.time()
    print(f"Finished precomputing STRIKE/BARRIER grids in {end_total_time - start_total_time:.2f} seconds.")
    return computed_grids, strike_values, barrier_values

# --- Plotting --- 
def create_interactive_heatmap(cfg, all_grids_data, strike_vals, barrier_vals):
    initial_params_tuple = (strike_vals[0], barrier_vals[0])
    initial_grid_info = all_grids_data.get(initial_params_tuple)

    if initial_grid_info is None:
        valid_key = next((k for k, v in all_grids_data.items() if v is not None), None)
        if valid_key:
            initial_params_tuple = valid_key
            initial_grid_info = all_grids_data[valid_key]
        else:
            print("Error: No valid precomputed grids for STRIKE/BARRIER. Cannot create heatmap.")
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
    # Slider 1: Strike
    steps_k = []
    for val in strike_vals:
        target_frame_name = get_frame_name((val, barrier_vals[0])) # Assumes barrier slider is at its initial position
        steps_k.append({
            "label": f"{val:.2f}", "method": "animate",
            "args": [[target_frame_name], {"mode": "immediate", "frame": {"duration": 50, "redraw": True}}]
        })
    sliders_definitions.append({
        "active": 0, "currentvalue": {"prefix": "Strike K: ", "visible": True, "xanchor": "right"},
        "pad": {"t": 20, "b": 10}, # Górne i dolne wypełnienie
        "steps": steps_k,
        "len": 0.8,  # Zwiększamy długość suwaka
        "x": 0.1,    # Wyrównujemy pozycję poziomą (10% od lewej)
        "y": -0.2    # Pozycja pionowa dla górnego suwaka (poniżej wykresu)
    })

    # Slider 2: Barrier
    steps_b = []
    for val in barrier_vals:
        target_frame_name = get_frame_name((strike_vals[0], val)) # Assumes strike slider is at its initial position
        steps_b.append({
            "label": f"{val:.2f}", "method": "animate",
            "args": [[target_frame_name], {"mode": "immediate", "frame": {"duration": 50, "redraw": True}}]
        })
    sliders_definitions.append({
        "active": 0, "currentvalue": {"prefix": "Barrier B: ", "visible": True, "xanchor": "right"},
        "pad": {"t": 60, "b": 10}, # Zwiększamy górne wypełnienie, aby zrobić miejsce pod pierwszym suwakiem
        "steps": steps_b,
        "len": 0.8,  # Taka sama długość jak górny suwak
        "x": 0.1,    # Taka sama pozycja pozioma jak górny suwak
        "y": -0.3    # Pozycja pionowa dla dolnego suwaka (niżej niż pierwszy)
    })

    updatemenus = [
        dict(type="buttons", direction="left", x=0.5, xanchor="center", y=-0.2, yanchor="top", showactive=False,
            buttons=[
                dict(label="Play All Frames", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ])]

    fig.update_layout(
        title=f"Interactive Heatmap (Strike/Barrier): {cfg['option_type']}<br><sup>VolMid={cfg['fixed_vol_mid']:.2f}, VolLen={cfg['fixed_vol_length']:.2f}, DivAmt={cfg['fixed_dividend_amount']}@T={cfg['fixed_dividend_time']}</sup>",
        xaxis_title="Time to Expiration (Years)", yaxis_title="Asset Price (S)",
        sliders=sliders_definitions, updatemenus=updatemenus,
        height=650 + len(sliders_definitions) * 40,
        width=800 + len(sliders_definitions) * 40
    )
    
    try:
        fig.write_html(generated_path(cfg["output_html_file"]))
        print(f"Interactive STRIKE/BARRIER heatmap saved to {cfg['output_html_file']}")
    except Exception as e:
        print(f"Error saving STRIKE/BARRIER heatmap: {e}")

if __name__ == "__main__":
    mp.freeze_support()
    config = get_config()
    print("Starting STRIKE/BARRIER heatmap generation...")
    all_grids, strike_v, barrier_v = precompute_all_grids(config)
    if not all_grids or all(grid_data is None for grid_data in all_grids.values()):
        print("No STRIKE/BARRIER grid data computed. Exiting.")
    else:
        print("--- Creating STRIKE/BARRIER Interactive Heatmap --- ")
        create_interactive_heatmap(config, all_grids, strike_v, barrier_v)
    print("--- STRIKE/BARRIER Script Finished ---")

