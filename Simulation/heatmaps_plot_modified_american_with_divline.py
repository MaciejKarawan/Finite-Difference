# Modified version of heatmaps_plot_modified_american.py
# Changes:
# 1. Assumes options are always barrier options with dividends.
# 2. Uses FD_KO_dividend_minimal.py for FDM calculations.
# 3. Calls imported plot_single_heatmap and plot_comparative_heatmap, 
#    assuming they are the versions that accept dividend line parameters.

"""
Generates static Early Execution heatmap plots for American options.
Assumes options are always barrier options with dividends.
Includes:
- 1x3 heatmaps for each option type (fixed volatilities).
- 1x2 comparative heatmaps for barrier options (fixed vs uncertain vol).
Assumes FD_KO_dividend_minimal.py, and the modified base plotting script
(containing updated plot_single_heatmap) are in the Python path.
Plots are saved in the current working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Ensure modules are in the same directory or Python path
try:
    # Use the FDM module that handles dividends for barrier options
    import FD_KO_dividend as fdm_dividend_module
    from matplotlib.colors import ListedColormap
    import seaborn as sns # Imported in the base, good to have here if used directly

    # This import should bring in the MODIFIED plot_single_heatmap and plot_comparative_heatmap
    # which now accept dividend line parameters. The user needs to ensure that the 
    # heatmaps_plot_modified.py (or whatever IF1.FD.heatmaps_plot_modified points to)
    # contains the updated plotting functions (like those in heatmaps_plot_modified_prices_with_divline.py)
    from heatmaps_plot_modified_prices_with_divline import plot_single_heatmap, plot_comparative_heatmap, setup_plotting_style
    
    print("Successfully imported fdm_dividend_module and plotting functions.")
except ImportError as e:
    print(f"Error importing user modules: {e}")
    print("Please ensure FD_KO_dividend_minimal.py and the modified base plotting script (e.g., heatmaps_plot_modified.py with updated plotting functions) are in your Python path.")
    sys.exit(1)

setup_plotting_style() # Call the imported setup

# --- Global Parameters ---
s0 = 2194.05
strike = 2200
expiration = 1.0
r = np.log(1 + 0.05)
fixed_vols_for_1x3_heatmap = [0.1, 0.25, 0.50]
vol_range_uncertain = [0.2, 0.3]
barrier_call = 2500
barrier_put = 1900
NAS_heatmap = 400

# --- Dividend Parameters (ASSUMED as per user request) ---
IS_DIVIDEND_ASSUMED = True
DIVIDEND_TIME_FRACTION_ASSUMED = 0.5 
DIVIDEND_VALUE_ASSUMED = 500

# --- Main Execution --- #
if __name__ == '__main__':
    plot_files = []

    # Option definitions for American options (barrier assumed)
    option_definitions = {
        "a_call_ko": ("American Call KO", barrier_call),
        "a_put_ko": ("American Put KO", barrier_put),
    }

    print("\n--- Generating American Option Execution Heatmap Visualizations (with Dividend Line) ---")
    cmap_binary = ListedColormap(["royalblue", "red"]) # For early execution

    # --- 1x3 Fixed Volatility Plots for Early Execution --- #
    print(f"Generating 1x3 Execution Heatmaps for Fixed Volatilities: {fixed_vols_for_1x3_heatmap}")

    for option_code, (option_name, barrier_val) in option_definitions.items():
        if barrier_val is None: continue # Should not happen with current defs

        print(f"\n  Processing: {option_name} ({option_code})")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f"{option_name}_{barrier_val} @{strike}, T_d = {DIVIDEND_TIME_FRACTION_ASSUMED}, D = {DIVIDEND_VALUE_ASSUMED}, Execution (NAS={NAS_heatmap})", fontsize=14)
        grids, S_vecs, time_vecs, plot_successful, ims = [], [], [], [False]*3, [None]*3

        for i, vol_fixed in enumerate(fixed_vols_for_1x3_heatmap):
            print(f"    Running with vol = {vol_fixed}...")
            try:
                vol_input = [vol_fixed, vol_fixed]
                base_option_type = option_code.split("_ko")[0]
                
                _, _, init_hm, grid_data_hm = fdm_dividend_module.FDB_with_memory(
                    NAS=NAS_heatmap, s0=s0, vol=vol_input, r=r,
                    option_type=base_option_type, strike=strike, expiration=expiration,
                    barrier=barrier_val, 
                    is_dividend=IS_DIVIDEND_ASSUMED,
                    dividend_time=DIVIDEND_TIME_FRACTION_ASSUMED, 
                    dividend_value=DIVIDEND_VALUE_ASSUMED
                )
                grid = grid_data_hm.get("Early_Exercise") # Get Early_Exercise grid
                S_vec, time_vec = init_hm.get("S"), init_hm.get("time_grid")
                if grid is not None and S_vec is not None and time_vec is not None:
                    grids.append(grid); S_vecs.append(S_vec); time_vecs.append(time_vec)
                    plot_successful[i] = True
                else: 
                    print(f"      Warning: Missing Early_Exercise data for vol={vol_fixed}.")
                    grids.append(None); S_vecs.append(None); time_vecs.append(None)
            except Exception as e:
                print(f"      Error generating Early_Exercise heatmap data for vol={vol_fixed}: {e}")
                grids.append(None); S_vecs.append(None); time_vecs.append(None)
        
        # For binary Early_Exercise data, vmin=0, vmax=1 is appropriate
        vmin_exec, vmax_exec = 0, 1 
        dividend_time_on_x_axis = DIVIDEND_TIME_FRACTION_ASSUMED * expiration if IS_DIVIDEND_ASSUMED else None

        for i, vol_fixed in enumerate(fixed_vols_for_1x3_heatmap):
            if plot_successful[i]:
                title = f"Vol={vol_fixed*100:.0f}%"
                ims[i] = plot_single_heatmap(axes[i], grids[i], S_vecs[i], time_vecs[i], title, 
                                             cmap=cmap_binary, vmin=vmin_exec, vmax=vmax_exec,
                                             strike_price_param=strike, 
                                             is_dividend_param=IS_DIVIDEND_ASSUMED,
                                             dividend_time_abs_param=dividend_time_on_x_axis, 
                                             current_expiration_param=expiration)
            else:
                axes[i].set_title(f"Vol={vol_fixed*100:.0f}% (Error)")
                axes[i].text(0.5, 0.5, "Calculation Failed", ha="center", va="center")
                axes[i].set_xlabel("Time to Expiration (Years)"); axes[i].set_ylabel("Asset Price (S)")
        
        plt.tight_layout(rect=[0, 0.03, 1.08, 0.95]) # Adjusted for colorbar
        im_for_colorbar = next((im for im in ims if im is not None), None)
        if im_for_colorbar: 
            cbar = fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label="Early Execution Decision", shrink=1)
            cbar.set_ticks([0, 1])
            cbar.ax.set_yticklabels(["Not Exercised", "Exercised"])
        
        fname = f"heatmap_1x3_execution_divline_{option_code}_vols_{fixed_vols_for_1x3_heatmap[0]}_{fixed_vols_for_1x3_heatmap[1]}_{fixed_vols_for_1x3_heatmap[2]}_dividend_{DIVIDEND_VALUE_ASSUMED}_{DIVIDEND_TIME_FRACTION_ASSUMED}.png"
        try:
            plt.savefig(fname, bbox_inches='tight', dpi=300, format='png'); plt.close(fig)
            print(f"  1x3 Execution Heatmap saved to {fname}"); plot_files.append(fname)
        except Exception as e: print(f"  Error saving 1x3 Execution heatmap {fname}: {e}"); plt.close(fig)

    # --- Comparative Heatmaps Section (Early Execution with Dividend Line) ---
    print("\nGenerating Comparative Execution Heatmaps: Barrier Options (Fixed vs Uncertain Volatility) - with Dividend Line")
    for option_code in option_definitions.keys(): 
        name_hm_barrier, barrier_hm_barrier = option_definitions[option_code]
        base_option_type_hm_barrier = option_code.split("_ko")[0]
        vol_mean_uncertain = np.mean(vol_range_uncertain)
        vol_fixed_for_barrier_comp = [vol_mean_uncertain, vol_mean_uncertain]
        print(f"\n  Processing Comparative Execution: {name_hm_barrier}")
        grid_fixed, S_vec_fixed, time_vec_fixed = None, None, None
        grid_unc, S_vec_unc, time_vec_unc = None, None, None
        try:
            _, _, init_f, grid_data_f = fdm_dividend_module.FDB_with_memory(NAS=NAS_heatmap, s0=s0, vol=vol_fixed_for_barrier_comp, r=r, option_type=base_option_type_hm_barrier, strike=strike, expiration=expiration, barrier=barrier_hm_barrier, is_dividend=IS_DIVIDEND_ASSUMED, dividend_time=DIVIDEND_TIME_FRACTION_ASSUMED, dividend_value=DIVIDEND_VALUE_ASSUMED)
            grid_fixed = grid_data_f.get("Early_Exercise")
            S_vec_fixed, time_vec_fixed = init_f.get("S"), init_f.get("time_grid")
        except Exception as e: print(f"      Error (Fixed Vol Execution): {e}")
        try:
            _, _, init_u, grid_data_u = fdm_dividend_module.FDB_with_memory(NAS=NAS_heatmap, s0=s0, vol=vol_range_uncertain, r=r, option_type=base_option_type_hm_barrier, strike=strike, expiration=expiration, barrier=barrier_hm_barrier, is_dividend=IS_DIVIDEND_ASSUMED, dividend_time=DIVIDEND_TIME_FRACTION_ASSUMED, dividend_value=DIVIDEND_VALUE_ASSUMED)
            grid_unc = grid_data_u.get("Early_Exercise")
            S_vec_unc, time_vec_unc = init_u.get("S"), init_u.get("time_grid")
        except Exception as e: print(f"      Error (Uncertain Vol Execution): {e}")

        if grid_fixed is not None and grid_unc is not None:
            title_fixed = f"Fixed Vol ({vol_mean_uncertain*100:.1f}%)"
            title_unc = f"Uncertain Vol ({vol_range_uncertain[0]*100:.0f}-{vol_range_uncertain[1]*100:.0f}%)"
            figure_title = f"{option_name}_{barrier_val} @{strike}, T_d = {DIVIDEND_TIME_FRACTION_ASSUMED}, D = {DIVIDEND_VALUE_ASSUMED}, Execution (NAS={NAS_heatmap})"
            fname_comp = f"heatmap_comparison_execution_divline_{option_code}_volrange_{vol_range_uncertain[0]}_{vol_range_uncertain[1]}_dividend_{DIVIDEND_VALUE_ASSUMED}_{DIVIDEND_TIME_FRACTION_ASSUMED}.png"
            dividend_time_on_x_axis_comp = DIVIDEND_TIME_FRACTION_ASSUMED * expiration if IS_DIVIDEND_ASSUMED else None
            plot_file_comp = plot_comparative_heatmap(
                grid_fixed, S_vec_fixed, time_vec_fixed, title_fixed, 
                grid_unc, S_vec_unc, time_vec_unc, title_unc, 
                figure_title, fname_comp, cmap=cmap_binary, label='Early Execution Decision',
                is_dividend_param=IS_DIVIDEND_ASSUMED, 
                dividend_time_abs_param=dividend_time_on_x_axis_comp, 
                current_expiration_param=expiration, 
                strike_price_param_comp=strike)
            if plot_file_comp: plot_files.append(plot_file_comp)
        else: print(f"    Skipping comparative Execution heatmap for {name_hm_barrier} due to missing data.")

    print("\n--- American Option Execution Heatmap Visualizations Generated ---")
    if plot_files: print("Generated files:"); [print(f"  {f}") for f in plot_files]
    else: print("No American Execution heatmap files were generated successfully.")
    print("\nScript finished.")

