from finite_difference.paths import generated_path
"""
Generates static heatmap plots for Finite Difference method analysis.
Includes:
- 1x3 heatmaps for each option type (fixed volatilities).
- 1x2 comparative heatmaps for barrier options (fixed vs uncertain vol).
Assumes FD.py, FD_KO.py, interpolation.py, and pricing.py are in the same directory or Python path.
Plots are saved in the current working directory.
"""



import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Ensure modules are in the same directory or Python path
try:
    from finite_difference import fd as FD
    from finite_difference import fd_ko as FD_KO
    from finite_difference import fd_ko_dividend as FD_KO_dividend
    from matplotlib.colors import ListedColormap
    import seaborn as sns
    # pricing and interpolation are used internally by FD and FD_KO
except ImportError as e:
    print(f"Error importing user modules: {e}")
    print("Please ensure FD.py, FD_KO.py, interpolation.py, and pricing.py are in the same directory as this script or in your Python path.")
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
s0 = 2194.05 # From user's provided script
strike = 2200
expiration = 1.0
r = np.log(1 + 0.05)
# Easily editable list for fixed volatility heatmaps
fixed_vols_for_1x3_heatmap = [0.1, 0.25, 0.50]
# Volatility range for barrier options (used for uncertain case in comparative plots)
vol_range_uncertain = [0.1, 0.5]
barrier_call = 2500
barrier_put = 1900
NAS_heatmap = 400      # NAS for heatmap generation

# --- Helper Functions ---

def plot_single_heatmap(ax, grid, S_vec, time_vec, title, cmap="inferno", vmin=None, vmax=None):
    """Generates a single heatmap on a given Axes object."""
    # --- Adjust Grid/Vector Dimensions if Necessary --- #
    if grid is None or S_vec is None or time_vec is None:
        print(f"Warning: Cannot plot heatmap for {title} due to missing data.")
        ax.text(0.5, 0.5, "Error: No data", horizontalalignment="center", verticalalignment="center")
        return None

    if grid.shape[0] == len(S_vec) - 1:
        S_vec_adjusted = S_vec[:-1]
    elif grid.shape[0] == len(S_vec):
        S_vec_adjusted = S_vec
    else:
        print(f"Warning: Heatmap grid shape {grid.shape} and S_vec length {len(S_vec)} mismatch for {title}. Adjusting S.")
        # Attempt a reasonable adjustment, might need refinement based on FD output specifics
        S_vec_adjusted = S_vec[:grid.shape[0]] if grid.shape[0] <= len(S_vec) else S_vec

    if grid.shape[1] != len(time_vec):
         print(f"Warning: Heatmap grid shape {grid.shape} and time_vec length {len(time_vec)} mismatch for {title}. Adjusting time.")
         # Attempt a reasonable adjustment
         time_vec_adjusted = time_vec[:grid.shape[1]] if grid.shape[1] <= len(time_vec) else time_vec
    else:
        time_vec_adjusted = time_vec

    # --- Plotting --- #
    im = None
    if len(S_vec_adjusted) > 0 and len(time_vec_adjusted) > 0 and grid.shape == (len(S_vec_adjusted), len(time_vec_adjusted)):
        T, S = np.meshgrid(time_vec_adjusted, S_vec_adjusted)
        im = ax.pcolormesh(T, S, grid, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        # Rysuj grubszą czarną linię przerywaną jako "obramowanie"
        ax.axhline(y=strike, color='black', linestyle='--', linewidth=1.5)
        # Rysuj cieńszą szarą linię przerywaną na wierzchu jako "wypełnienie"
        ax.axhline(y=strike, color='grey', linestyle='--', linewidth=0.75)

    elif len(S_vec_adjusted) > 0 and len(time_vec_adjusted) > 0:
         print(f"Warning: Final dimension mismatch after adjustment for {title}. Grid: {grid.shape}, S: {len(S_vec_adjusted)}, T: {len(time_vec_adjusted)}. Attempting plot anyway.")
         # Try plotting with original vectors if adjustment failed
         try:
             T, S = np.meshgrid(time_vec, S_vec[:grid.shape[0]]) # Best guess
             im = ax.pcolormesh(T, S, grid, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
         except Exception as plot_err:
             print(f"    Plotting failed for {title}: {plot_err}")
             ax.text(0.5, 0.5, "Error: Plot failed", horizontalalignment="center", verticalalignment="center")
             return None
    else:
        print(f"Warning: Cannot plot heatmap for {title} due to zero dimension adjusted vectors.")
        ax.text(0.5, 0.5, "Error: No data", horizontalalignment="center", verticalalignment="center")
        return None

        # Pobierz aktualne znaczniki osi Y
    y_ticks = ax.get_yticks()
    # Dodaj strike do listy znaczników, jeśli go tam nie ma
    if strike not in y_ticks:
        y_ticks = np.append(y_ticks, strike)
        y_ticks.sort() # Posortuj, aby zachować porządek
        ax.set_yticks(y_ticks)

    # --- Labels and Title --- #
    ax.set_xlabel("Time to Expiration (Years)")
    ax.set_ylabel("Asset Price (S)")
    ax.set_title(title)
    return im # Return the image object for colorbar

def plot_comparative_heatmap(grid1, S_vec1, time_vec1, title1,
                             grid2, S_vec2, time_vec2, title2,
                             figure_title, filename, cmap='inferno', label='Option Price'):
    """Generates a 1x2 comparative heatmap plot (for uncertain vol section)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True) # Share Y axis
    fig.suptitle(figure_title, fontsize=14)

    # Determine common color scale limits
    valid_data1 = grid1[~np.isnan(grid1)] if grid1 is not None else np.array([])
    valid_data2 = grid2[~np.isnan(grid2)] if grid2 is not None else np.array([])
    vmin = min(valid_data1.min() if len(valid_data1)>0 else np.inf, valid_data2.min() if len(valid_data2)>0 else np.inf)
    vmax = max(valid_data1.max() if len(valid_data1)>0 else -np.inf, valid_data2.max() if len(valid_data2)>0 else -np.inf)
    if vmin == np.inf or vmax == -np.inf:
        vmin, vmax = None, None # Use default scaling if no valid data

    # Plot left heatmap (fixed vol)
    im1 = plot_single_heatmap(axes[0], grid1, S_vec1, time_vec1, title1, cmap, vmin=vmin, vmax=vmax)

    # Plot right heatmap (uncertain vol)
    im2 = plot_single_heatmap(axes[1], grid2, S_vec2, time_vec2, title2, cmap, vmin=vmin, vmax=vmax)

    # Adjust layout *before* adding colorbar to make space
    plt.tight_layout(rect=[0, 0.03, 1.11, 0.95]) # Leave space on the right (rect=[left, bottom, right, top])

    # Add a single colorbar for the whole figure
    if label not in ['Early Execution Decision', 'Gamma sign']:
        if im1 or im2:
            # Use the image object that was successfully created
            im_for_colorbar = im1 if im1 is not None else im2
            if im_for_colorbar:
                 fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label=label, shrink=1)
    elif label == 'Early Execution Decision':
        # Ustaw etykietę w zależności od typu opcji
        cbar_label = "Early Execution Decision"
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), label=cbar_label, shrink=1)
        cbar.set_ticks([0, 1])
        cbar.ax.set_yticklabels(["Not Exercised", "Exercised"])
    else:
        cbar_label = "Gamma sign"
        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), label=cbar_label, shrink=1)
        cbar.set_ticks([0, 1])
        cbar.ax.set_yticklabels([r'$\Gamma < 0$ -> $\sigma_-$', r'$\Gamma \geq 0$ -> $\sigma_+$'])
    # --- Save Plot --- #
    filepath = filename # Save directly with the filename in the current directory
    try:
        plt.savefig(generated_path(filepath), bbox_inches='tight', dpi=300, format='png')
        plt.close(fig)
        print(f"Comparative heatmap saved to {filepath}")
    except Exception as e:
        print(f"Error saving comparative heatmap {filepath}: {e}")
        plt.close(fig)
    return filepath

if __name__ == '__main__':

    # --- Main Execution --- #

    plot_files = []

    # Define option types and their properties
    option_definitions = {
        # Code: (Display Name, Barrier Value or None)
        "e_call": ("European Call", None),
        "e_put": ("European Put", None),
        "a_call": ("American Call", None),
        "a_put": ("American Put", None),
        "e_call_ko": ("European Call KO", barrier_call),
        "e_put_ko": ("European Put KO", barrier_put),
        "a_call_ko": ("American Call KO", barrier_call),
        "a_put_ko": ("American Put KO", barrier_put),
    }
    '''
    # == Heatmap Visualizations ==
    print("\n--- Generating Heatmap Visualizations ---")

    # --- MODIFIED SECTION: Heatmaps for ALL Options (1x3 Fixed Volatility Plots) --- #
    print(f"Generating 1x3 Heatmaps for Fixed Volatilities: {fixed_vols_for_1x3_heatmap}")

    for option_code, (option_name, barrier_val) in option_definitions.items():
        print(f"\n  Processing: {option_name} ({option_code})")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True) # 1 row, 3 columns

        if barrier_val is None:
            fig.suptitle(f"{option_name} @{strike} Price (NAS={NAS_heatmap})", fontsize=14)
        else:
            fig.suptitle(f"{option_name}_{barrier_val} @{strike} Price (NAS={NAS_heatmap})", fontsize=14)

        grids = []
        S_vecs = []
        time_vecs = []
        plot_successful = [False] * 3
        ims = [None] * 3

        # Calculate data for each volatility
        for i, vol_fixed in enumerate(fixed_vols_for_1x3_heatmap):
            print(f"    Running with vol = {vol_fixed}...")
            grid = None
            S_vec = None
            time_vec = None
            try:
                if barrier_val is None: # Non-barrier option
                    _, _, init_hm, grid_data_hm = FD.FD_with_memory(
                        NAS=NAS_heatmap,
                        s0=s0,
                        vol=vol_fixed,
                        r=r,
                        option_type=option_code,
                        strike=strike,
                        expiration=expiration
                    )
                else: # Barrier option
                    # FDB_with_memory expects vol as a list [vol, vol] for fixed case
                    vol_input = [vol_fixed, vol_fixed]
                    base_option_type = option_code.split("_ko")[0]
                    _, _, init_hm, grid_data_hm = FD_KO.FDB_with_memory(
                        NAS=NAS_heatmap,
                        s0=s0,
                        vol=vol_input,
                        r=r,
                        option_type=base_option_type,
                        strike=strike,
                        expiration=expiration,
                        barrier=barrier_val
                    )

                grid = grid_data_hm.get("Grid")
                S_vec = init_hm.get("S")
                time_vec = init_hm.get("time_grid")

                if grid is not None and S_vec is not None and time_vec is not None:
                    grids.append(grid)
                    S_vecs.append(S_vec)
                    time_vecs.append(time_vec)
                    plot_successful[i] = True # Mark as potentially plottable
                else:
                    print(f"      Warning: Missing data for vol={vol_fixed}.")
                    grids.append(None) # Keep lists aligned
                    S_vecs.append(None)
                    time_vecs.append(None)

            except Exception as e:
                print(f"      Error generating heatmap data for vol={vol_fixed}: {e}")
                grids.append(None)
                S_vecs.append(None)
                time_vecs.append(None)

        # Determine common color scale limits across the 3 plots for this option type
        all_valid_data = []
        for g in grids:
            if g is not None:
                all_valid_data.extend(g[~np.isnan(g)].flatten())

        vmin = np.min(all_valid_data) if all_valid_data else None
        vmax = np.max(all_valid_data) if all_valid_data else None

        # Plot each heatmap
        for i, vol_fixed in enumerate(fixed_vols_for_1x3_heatmap):
            if plot_successful[i]:
                title = f"Vol={vol_fixed*100:.0f}%"
                ims[i] = plot_single_heatmap(axes[i], grids[i], S_vecs[i], time_vecs[i], title, vmin=vmin, vmax=vmax)
            else:
                # Handle cases where calculation failed
                axes[i].set_title(f"Vol={vol_fixed*100:.0f}% (Error)")
                axes[i].text(0.5, 0.5, "Calculation Failed", horizontalalignment="center", verticalalignment="center")
                axes[i].set_xlabel("Time to Expiration (Years)")
                axes[i].set_ylabel("Asset Price (S)")

        # Add a single colorbar for the figure
        # Find the first successful plot to use for the colorbar
        plt.tight_layout(rect=[0, 0.03, 1.11, 0.95]) # Leave space on the right (rect=[left, bottom, right, top])
        im_for_colorbar = next((im for im in ims if im is not None), None)
        if im_for_colorbar:
            fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label="Option Price", shrink=1)

        # Adjust layout and save
        #plt.tight_layout(rect=[0, 0.03, 0.95, 0.95]) # Adjust rect to prevent title overlap and make space for colorbar
        fname = f"heatmap_1x3_fixed_vol_{option_code}_vols_{fixed_vols_for_1x3_heatmap[0]}_{fixed_vols_for_1x3_heatmap[1]}_{fixed_vols_for_1x3_heatmap[2]}.png"
        filepath = fname
        try:
            plt.savefig(filepath, bbox_inches='tight', dpi=300, format='png')
            plt.close(fig)
            print(f"  1x3 Heatmap saved to {filepath}")
            plot_files.append(filepath)
        except Exception as e:
            print(f"  Error saving 1x3 heatmap {filepath}: {e}")
            plt.close(fig)
    '''

    # --- UNCHANGED SECTION: Heatmaps for Barrier Options (Comparative 1x2 Plots) --- #
    print("\nGenerating Heatmaps: Barrier Options (Fixed vs Uncertain Volatility)")

    # Loop through barrier options defined in option_definitions
    barrier_option_codes = [code for code, (_, barrier_val) in option_definitions.items() if barrier_val is not None]

    for option_code in barrier_option_codes:
        name_hm_barrier, barrier_hm_barrier = option_definitions[option_code]
        base_option_type_hm_barrier = option_code.split("_ko")[0]

        # Calculate mean volatility for the 'fixed' part of the comparison
        vol_mean_uncertain = np.mean(vol_range_uncertain)
        vol_fixed_for_barrier_comp = [vol_mean_uncertain, vol_mean_uncertain] # Pass as list [mean, mean]

        print(f"\n  Processing Comparative: {name_hm_barrier}")
        print(f"    Running with Fixed Vol ({vol_mean_uncertain*100:.1f}%) and Uncertain Vol ({vol_range_uncertain})...")

        grid_fixed, S_vec_fixed, time_vec_fixed = None, None, None
        grid_unc, S_vec_unc, time_vec_unc = None, None, None

        # 1. Calculate for Fixed Volatility (Mean of uncertain range)
        try:
            print(f"      Calculating Fixed Vol for comparison... ({vol_fixed_for_barrier_comp})")
            _, _, init_fixed, grid_data_fixed = FD_KO.FDB_with_memory(
                NAS=NAS_heatmap,
                s0=s0,
                vol=vol_fixed_for_barrier_comp, # Use [mean, mean]
                r=r,
                option_type=base_option_type_hm_barrier,
                strike=strike,
                expiration=expiration,
                barrier=barrier_hm_barrier
            )
            grid_fixed = grid_data_fixed.get("Grid")
            S_vec_fixed = init_fixed.get("S")
            time_vec_fixed = init_fixed.get("time_grid")
        except Exception as e:
            print(f"      Error generating fixed vol comparative heatmap data: {e}")

        # 2. Calculate for Uncertain Volatility (Range)
        try:
            print(f"      Calculating Uncertain Vol... ({vol_range_uncertain})")
            _, _, init_unc, grid_data_unc = FD_KO.FDB_with_memory(
                NAS=NAS_heatmap,
                s0=s0,
                vol=vol_range_uncertain, # Use full range
                r=r,
                option_type=base_option_type_hm_barrier,
                strike=strike,
                expiration=expiration,
                barrier=barrier_hm_barrier
            )
            grid_unc = grid_data_unc.get("Grid")
            S_vec_unc = init_unc.get("S") # Should be same as S_vec_fixed
            time_vec_unc = init_unc.get("time_grid") # Should be same as time_vec_fixed
        except Exception as e:
            print(f"      Error generating uncertain vol comparative heatmap data: {e}")

        # 3. Plot Comparative Heatmap if data is available
        if grid_fixed is not None and S_vec_fixed is not None and time_vec_fixed is not None and \
                grid_unc is not None and S_vec_unc is not None and time_vec_unc is not None:

            title_fixed = f"Fixed Vol ({vol_mean_uncertain*100:.1f}%)"
            title_unc = f"Uncertain Vol ({vol_range_uncertain[0]*100:.0f}-{vol_range_uncertain[1]*100:.0f}%)"
            figure_title = f"{name_hm_barrier}_{barrier_hm_barrier} @{strike} (NAS={NAS_heatmap})"
            fname_comp = f"heatmap_comparison_{option_code}_vol_range_{vol_range_uncertain[0]}_{vol_range_uncertain[1]}.png"

            print(f"    Plotting comparative heatmap...")
            plot_file_comp = plot_comparative_heatmap(
                grid_fixed, S_vec_fixed, time_vec_fixed, title_fixed,
                grid_unc, S_vec_unc, time_vec_unc, title_unc,
                figure_title, fname_comp
            )
            if plot_file_comp:
                 plot_files.append(plot_file_comp)
        else:
            print(f"    Skipping comparative heatmap for {name_hm_barrier} due to missing data.")


    print("\n--- Heatmap Visualizations Generated ---")
    heatmap_files_generated = [f for f in plot_files if f and "heatmap" in os.path.basename(f)]
    if heatmap_files_generated:
        print("Generated files:")
        for f in heatmap_files_generated:
            print(f"  {f}")
    else:
        print("No heatmap files were generated successfully.")

    print("\nScript finished.")

