from finite_difference.paths import generated_path
# Modified version of heatmaps_plot_modified_gamma.py
# Changes:
# 1. Assumes options are always barrier options with dividends.
# 2. Uses FD_KO_dividend_minimal.py for FDM calculations.
# 3. plot_single_heatmap and plot_comparative_heatmap updated to draw dividend line.

"""
Generates static Gamma heatmap plots for Finite Difference method analysis.
Includes:
- 1x3 heatmaps for each option type (fixed volatilities).
- 1x2 comparative heatmaps for barrier options (fixed vs uncertain vol).
Assumes FD_KO_dividend_minimal.py, interpolation.py, and pricing.py are in the same directory or Python path.
Plots are saved in the current working directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Ensure modules are in the same directory or Python path
try:
    from finite_difference import fd_ko_dividend as fdm_dividend_module
    from matplotlib.colors import ListedColormap 
    import seaborn as sns
    print("Successfully imported fdm_dividend_module (FD_KO_dividend_minimal.py)")
except ImportError as e:
    print(f"Error importing user modules: {e}")
    print("Please ensure FD_KO_dividend_minimal.py and its dependencies (interpolation.py, pricing.py) are in the same directory as this script or in your Python path.")
    sys.exit(1)

def setup_plotting_style():
    """Konfiguruje styl wykresów dla całego projektu."""
    plt.rc("figure",
           autolayout=True,
           figsize=(11.69, 8.27),
           titlesize=10,
           titleweight='bold')
    plt.rc("axes",
           labelweight="bold",
           labelsize="large",
           titleweight="bold",
           titlesize=16,
           titlepad=10,
           titlelocation='center')
    sns.set_theme(style='whitegrid', context='paper', palette='dark')
setup_plotting_style()

# --- Global Parameters ---
s0 = 2194.05
strike = 2200
expiration = 1.0
r = np.log(1 + 0.05)
fixed_vols_for_1x3_heatmap = [0.1, 0.25, 0.50]
vol_range_uncertain = [0.2, 0.3]
barrier_call = 2500
barrier_put = 1900
NAS_heatmap = 100

# --- Dividend Parameters (ASSUMED as per user request) ---
IS_DIVIDEND_ASSUMED = True
DIVIDEND_TIME_FRACTION_ASSUMED = 0.5 
DIVIDEND_VALUE_ASSUMED = 300

# --- Helper Functions (Copied and modified from heatmaps_plot_modified_prices_with_divline.py) ---
def plot_single_heatmap(ax, grid, S_vec, time_vec, title, cmap="inferno", vmin=None, vmax=None, 
                        strike_price_param=None, 
                        is_dividend_param=False, 
                        dividend_time_abs_param=None, 
                        current_expiration_param=None):
    """Generates a single heatmap on a given Axes object."""
    if grid is None or S_vec is None or time_vec is None:
        print(f"Warning: Cannot plot heatmap for {title} due to missing data.")
        ax.text(0.5, 0.5, "Error: No data", horizontalalignment="center", verticalalignment="center")
        return None

    S_vec_adjusted = S_vec
    if grid.shape[0] == len(S_vec) - 1:
        S_vec_adjusted = S_vec[:-1]
    elif grid.shape[0] != len(S_vec):
        print(f"Warning: Heatmap grid shape {grid.shape} and S_vec length {len(S_vec)} mismatch for {title}. Adjusting S.")
        S_vec_adjusted = S_vec[:grid.shape[0]] if grid.shape[0] <= len(S_vec) else S_vec

    time_vec_adjusted = time_vec
    if grid.shape[1] != len(time_vec):
        print(f"Warning: Heatmap grid shape {grid.shape} and time_vec length {len(time_vec)} mismatch for {title}. Adjusting time.")
        time_vec_adjusted = time_vec[:grid.shape[1]] if grid.shape[1] <= len(time_vec) else time_vec

    im = None
    if len(S_vec_adjusted) > 0 and len(time_vec_adjusted) > 0 and grid.shape == (len(S_vec_adjusted), len(time_vec_adjusted)):
        T, S_mesh = np.meshgrid(time_vec_adjusted, S_vec_adjusted)
        im = ax.pcolormesh(T, S_mesh, grid, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        
        if strike_price_param is not None:
            ax.axhline(y=strike_price_param, color='black', linestyle='--', linewidth=1.5, zorder=5)
            ax.axhline(y=strike_price_param, color='grey', linestyle='--', linewidth=0.75, zorder=5)
            y_ticks = list(ax.get_yticks())
            if strike_price_param not in y_ticks:
                y_ticks.append(strike_price_param)
                y_ticks.sort()
                ax.set_yticks(y_ticks)

        if is_dividend_param and dividend_time_abs_param is not None and current_expiration_param is not None:
            min_time_plot = time_vec_adjusted.min()
            max_time_plot = time_vec_adjusted.max()
            if min_time_plot <= dividend_time_abs_param <= max_time_plot:
                ax.axvline(x=dividend_time_abs_param, color='black', linestyle=':', linewidth=1.5, zorder=5)
                ax.axvline(x=dividend_time_abs_param, color='darkorange', linestyle=':', linewidth=0.75, zorder=5)
                x_ticks = list(ax.get_xticks())
                is_close_to_existing_tick = any(abs(tick - dividend_time_abs_param) < (max_time_plot / 20) for tick in x_ticks)
                if not is_close_to_existing_tick:
                     x_ticks.append(dividend_time_abs_param)
                     x_ticks.sort()
                     ax.set_xticks(x_ticks)
                     ax.tick_params(axis='x', labelrotation=45)
            else:
                print(f"Warning: Dividend time {dividend_time_abs_param} outside plot x-axis range for {title}")

    elif len(S_vec_adjusted) > 0 and len(time_vec_adjusted) > 0:
        print(f"Warning: Final dimension mismatch for {title}. Attempting plot.")
        try:
            T, S_mesh = np.meshgrid(time_vec, S_vec[:grid.shape[0]])
            im = ax.pcolormesh(T, S_mesh, grid, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        except Exception as plot_err:
            print(f"    Plotting failed for {title}: {plot_err}")
            ax.text(0.5, 0.5, "Error: Plot failed", horizontalalignment="center", verticalalignment="center")
            return None
    else:
        print(f"Warning: Cannot plot heatmap for {title} due to zero dim vectors.")
        ax.text(0.5, 0.5, "Error: No data", horizontalalignment="center", verticalalignment="center")
        return None

    ax.set_xlabel("Time to Expiration (Years)")
    ax.set_ylabel("Asset Price (S)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    return im

def plot_comparative_heatmap(grid1, S_vec1, time_vec1, title1,
                             grid2, S_vec2, time_vec2, title2,
                             figure_title, filename, cmap='inferno', label='Option Price',
                             is_dividend_param=False, dividend_time_abs_param=None,
                             current_expiration_param=None, strike_price_param_comp=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(figure_title, fontsize=14)
    valid_data1 = grid1[~np.isnan(grid1)] if grid1 is not None else np.array([])
    valid_data2 = grid2[~np.isnan(grid2)] if grid2 is not None else np.array([])
    vmin = min(valid_data1.min() if len(valid_data1)>0 else np.inf, valid_data2.min() if len(valid_data2)>0 else np.inf)
    vmax = max(valid_data1.max() if len(valid_data1)>0 else -np.inf, valid_data2.max() if len(valid_data2)>0 else -np.inf)
    if vmin == np.inf or vmax == -np.inf: vmin, vmax = None, None

    im1 = plot_single_heatmap(axes[0], grid1, S_vec1, time_vec1, title1, cmap, vmin=vmin, vmax=vmax,
                              strike_price_param=strike_price_param_comp, is_dividend_param=is_dividend_param,
                              dividend_time_abs_param=dividend_time_abs_param, current_expiration_param=current_expiration_param)
    im2 = plot_single_heatmap(axes[1], grid2, S_vec2, time_vec2, title2, cmap, vmin=vmin, vmax=vmax,
                              strike_price_param=strike_price_param_comp, is_dividend_param=is_dividend_param,
                              dividend_time_abs_param=dividend_time_abs_param, current_expiration_param=current_expiration_param)

    plt.tight_layout(rect=[0, 0.03, 1.11, 0.95])
    im_for_colorbar = im1 if im1 is not None else im2
    if im_for_colorbar:
        if label not in ["Early Execution Decision", "Gamma sign"]:
            fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label=label, shrink=1)
        elif label == "Early Execution Decision":
            cbar = fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label=label, shrink=1)
            cbar.set_ticks([0, 1])
            cbar.ax.set_yticklabels(["Not Exercised", "Exercised"])
        elif label == "Gamma sign":
            cbar = fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label=label, shrink=1)
            cbar.set_ticks([0, 1])
            cbar.ax.set_yticklabels([r'$\Gamma < 0$ -> $\sigma_-$', r'$\Gamma \geq 0$ -> $\sigma_+$'])
    try:
        plt.savefig(generated_path(filename), bbox_inches='tight', dpi=300, format='png')
        plt.close(fig)
        print(f"Comparative heatmap saved to {filename}")
    except Exception as e:
        print(f"Error saving comparative heatmap {filename}: {e}")
        plt.close(fig)
    return filename

if __name__ == '__main__':
    plot_files = []
    option_definitions = {
        "e_call_ko": ("European Call KO", barrier_call),
        "e_put_ko": ("European Put KO", barrier_put),
        "a_call_ko": ("American Call KO", barrier_call),
        "a_put_ko": ("American Put KO", barrier_put),
    }
    print("\n--- Generating Gamma Heatmap Visualizations (with Dividend Line) ---")
    cmap_gamma_sign = ListedColormap(["#f7a49c", "#ff1717",]) # From user original gamma script
    print(f"Generating 1x3 Gamma Heatmaps for Fixed Volatilities: {fixed_vols_for_1x3_heatmap}")

    for option_code, (option_name, barrier_val) in option_definitions.items():
        if barrier_val is None: continue
        print(f"\n  Processing: {option_name} ({option_code})")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f"{option_name}_{barrier_val} @{strike} Gamma (NAS={NAS_heatmap})", fontsize=14)
        grids, S_vecs, time_vecs, plot_successful, ims = [], [], [], [False]*3, [None]*3

        for i, vol_fixed in enumerate(fixed_vols_for_1x3_heatmap):
            print(f"    Running with vol = {vol_fixed}...")
            try:
                vol_input = [vol_fixed, vol_fixed]
                base_option_type = option_code.split("_ko")[0]
                _, _, init_hm, grid_data_hm = fdm_dividend_module.FDB_with_memory(
                    NAS=NAS_heatmap, s0=s0, vol=vol_input, r=r,
                    option_type=base_option_type, strike=strike, expiration=expiration,
                    barrier=barrier_val, is_dividend=IS_DIVIDEND_ASSUMED,
                    dividend_time=DIVIDEND_TIME_FRACTION_ASSUMED, dividend_value=DIVIDEND_VALUE_ASSUMED
                )
                gamma_grid = grid_data_hm.get("Gamma")
                S_vec, time_vec = init_hm.get("S"), init_hm.get("time_grid")
                if gamma_grid is not None and S_vec is not None and time_vec is not None:
                    processed_gamma_grid = (gamma_grid >= 0).astype(int)
                    grids.append(processed_gamma_grid); S_vecs.append(S_vec); time_vecs.append(time_vec)
                    plot_successful[i] = True
                else: 
                    print(f"      Warning: Missing Gamma data for vol={vol_fixed}.")
                    grids.append(None); S_vecs.append(None); time_vecs.append(None)
            except Exception as e:
                print(f"      Error generating Gamma heatmap data for vol={vol_fixed}: {e}")
                grids.append(None); S_vecs.append(None); time_vecs.append(None)
        
        # For binary Gamma sign data, vmin=0, vmax=1 is appropriate
        vmin_gamma, vmax_gamma = 0, 1 
        dividend_time_on_x_axis = DIVIDEND_TIME_FRACTION_ASSUMED * expiration if IS_DIVIDEND_ASSUMED else None

        for i, vol_fixed in enumerate(fixed_vols_for_1x3_heatmap):
            if plot_successful[i]:
                title = f"Vol={vol_fixed*100:.0f}%"
                ims[i] = plot_single_heatmap(axes[i], grids[i], S_vecs[i], time_vecs[i], title, 
                                             cmap=cmap_gamma_sign, vmin=vmin_gamma, vmax=vmax_gamma,
                                             strike_price_param=strike, 
                                             is_dividend_param=IS_DIVIDEND_ASSUMED,
                                             dividend_time_abs_param=dividend_time_on_x_axis, 
                                             current_expiration_param=expiration)
            else:
                axes[i].set_title(f"Vol={vol_fixed*100:.0f}% (Error)")
                axes[i].text(0.5, 0.5, "Calculation Failed", ha="center", va="center")
                axes[i].set_xlabel("Time to Expiration (Years)"); axes[i].set_ylabel("Asset Price (S)")
        
        plt.tight_layout(rect=[0, 0.03, 1.11, 0.95])
        im_for_colorbar = next((im for im in ims if im is not None), None)
        if im_for_colorbar: 
            cbar = fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label="Gamma Sign", shrink=1)
            cbar.set_ticks([0, 1])
            cbar.ax.set_yticklabels([r'$\Gamma < 0$ -> $\sigma_-$', r'$\Gamma \geq 0$ -> $\sigma_+$']) # Consistent with Gamma script
        
        fname = f"heatmap_1x3_gamma_divline_{option_code}_vols_{fixed_vols_for_1x3_heatmap[0]}_{fixed_vols_for_1x3_heatmap[1]}_{fixed_vols_for_1x3_heatmap[2]}_dividend_{DIVIDEND_VALUE_ASSUMED}_{DIVIDEND_TIME_FRACTION_ASSUMED}.png"
        try:
            plt.savefig(generated_path(fname), bbox_inches='tight', dpi=300, format='png'); plt.close(fig)
            print(f"  1x3 Gamma Heatmap saved to {fname}"); plot_files.append(fname)
        except Exception as e: print(f"  Error saving 1x3 Gamma heatmap {fname}: {e}"); plt.close(fig)

    # --- Comparative Heatmaps Section (Gamma with Dividend Line) ---
    print("\nGenerating Comparative Gamma Heatmaps: Barrier Options (Fixed vs Uncertain Volatility) - with Dividend Line")
    for option_code in option_definitions.keys():
        name_hm_barrier, barrier_hm_barrier = option_definitions[option_code]
        base_option_type_hm_barrier = option_code.split("_ko")[0]
        vol_mean_uncertain = np.mean(vol_range_uncertain)
        vol_fixed_for_barrier_comp = [vol_mean_uncertain, vol_mean_uncertain]
        print(f"\n  Processing Comparative Gamma: {name_hm_barrier}")
        grid_fixed, S_vec_fixed, time_vec_fixed = None, None, None
        grid_unc, S_vec_unc, time_vec_unc = None, None, None
        try:
            _, _, init_f, grid_data_f = fdm_dividend_module.FDB_with_memory(NAS=NAS_heatmap, s0=s0, vol=vol_fixed_for_barrier_comp, r=r, option_type=base_option_type_hm_barrier, strike=strike, expiration=expiration, barrier=barrier_hm_barrier, is_dividend=IS_DIVIDEND_ASSUMED, dividend_time=DIVIDEND_TIME_FRACTION_ASSUMED, dividend_value=DIVIDEND_VALUE_ASSUMED)
            gamma_grid_fixed = grid_data_f.get("Gamma")
            if gamma_grid_fixed is not None: grid_fixed = (gamma_grid_fixed >=0).astype(int)
            S_vec_fixed, time_vec_fixed = init_f.get("S"), init_f.get("time_grid")
        except Exception as e: print(f"      Error (Fixed Vol Gamma): {e}")
        try:
            _, _, init_u, grid_data_u = fdm_dividend_module.FDB_with_memory(NAS=NAS_heatmap, s0=s0, vol=vol_range_uncertain, r=r, option_type=base_option_type_hm_barrier, strike=strike, expiration=expiration, barrier=barrier_hm_barrier, is_dividend=IS_DIVIDEND_ASSUMED, dividend_time=DIVIDEND_TIME_FRACTION_ASSUMED, dividend_value=DIVIDEND_VALUE_ASSUMED)
            gamma_grid_unc = grid_data_u.get("Gamma")
            if gamma_grid_unc is not None: grid_unc = (gamma_grid_unc >=0).astype(int)
            S_vec_unc, time_vec_unc = init_u.get("S"), init_u.get("time_grid")
        except Exception as e: print(f"      Error (Uncertain Vol Gamma): {e}")

        if grid_fixed is not None and grid_unc is not None:
            title_fixed = f"Fixed Vol ({vol_mean_uncertain*100:.1f}%)"
            title_unc = f"Uncertain Vol ({vol_range_uncertain[0]*100:.0f}-{vol_range_uncertain[1]*100:.0f}%)"
            figure_title = f"{name_hm_barrier}_{barrier_hm_barrier} @{strike} Gamma Sign (NAS={NAS_heatmap})"
            fname_comp = f"heatmap_comparison_gamma_divline_{option_code}_volrange_{vol_range_uncertain[0]}_{vol_range_uncertain[1]}_dividend_{DIVIDEND_VALUE_ASSUMED}_{DIVIDEND_TIME_FRACTION_ASSUMED}.png"
            dividend_time_on_x_axis_comp = DIVIDEND_TIME_FRACTION_ASSUMED * expiration if IS_DIVIDEND_ASSUMED else None
            plot_file_comp = plot_comparative_heatmap(
                grid_fixed, S_vec_fixed, time_vec_fixed, title_fixed, 
                grid_unc, S_vec_unc, time_vec_unc, title_unc, 
                figure_title, fname_comp, cmap=cmap_gamma_sign, label='Gamma sign',
                is_dividend_param=IS_DIVIDEND_ASSUMED, 
                dividend_time_abs_param=dividend_time_on_x_axis_comp, 
                current_expiration_param=expiration, 
                strike_price_param_comp=strike)
            if plot_file_comp: plot_files.append(plot_file_comp)
        else: print(f"    Skipping comparative Gamma heatmap for {name_hm_barrier} due to missing data.")

    print("\n--- Gamma Heatmap Visualizations Generated ---")
    if plot_files: print("Generated files:"); [print(f"  {f}") for f in plot_files]
    else: print("No Gamma heatmap files were generated successfully.")
    print("\nScript finished.")

