from finite_difference.paths import generated_path
"""
Generates static heatmap plots for Finite Difference method analysis.
Includes:
- 1x3 heatmaps for each option type (fixed volatilities).
- 1x2 comparative heatmaps for barrier options (fixed vs uncertain vol).
Assumes FD.py, FD_KO.py, interpolation.py, and pricing.py are in the same directory or Python path.
Plots are saved in the current working directory.
"""

import sys

# Ensure modules are in the same directory or Python path
try:
    from finite_difference import fd as FD
    from finite_difference import fd_ko as FD_KO
    from scripts.plots.heatmaps_plot_modified import *
    # pricing and interpolation are used internally by FD and FD_KO
except ImportError as e:
    print(f"Error importing user modules: {e}")
    print("Please ensure FD.py, FD_KO.py, interpolation.py, and pricing.py are in the same directory as this script or in your Python path.")
    sys.exit(1)

# --- Global Parameters ---
s0 = 2194.05 # From user's provided script
strike = 2200
expiration = 1.0
r = np.log(1 + 0.05)
# Easily editable list for fixed volatility heatmaps
fixed_vols_for_1x3_heatmap = [0.1, 0.25, 0.50]
# Volatility range for barrier options (used for uncertain case in comparative plots)
vol_range_uncertain = [0.2, 0.3]
barrier_call = 2500
barrier_put = 1900
NAS_heatmap = 400      # NAS for heatmap generation


# --- Main Execution --- #

plot_files = []

# Define option types and their properties
option_definitions = {
    # Code: (Display Name, Barrier Value or None)
    #"e_call": ("European Call", None),
    #"e_put": ("European Put", None),
    #"a_call": ("American Call", None),
    #"a_put": ("American Put", None),
    "e_call_ko": ("European Call KO", barrier_call),
    "e_put_ko": ("European Put KO", barrier_put),
    "a_call_ko": ("American Call KO", barrier_call),
    "a_put_ko": ("American Put KO", barrier_put),
}

# == Heatmap Visualizations ==
print("\n--- Generating Gamma Heatmap Visualizations ---")

cmap_gamma_sign = ListedColormap(["#f7a49c", "#ff1717",])


# --- MODIFIED SECTION: Heatmaps for ALL Options (1x3 Fixed Volatility Plots) --- #
print(f"Generating 1x3 Gamma Heatmaps for Fixed Volatilities: {fixed_vols_for_1x3_heatmap}")

for option_code, (option_name, barrier_val) in option_definitions.items():
    print(f"\n  Processing: {option_name} ({option_code})")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True) # 1 row, 3 columns

    if barrier_val is None:
        fig.suptitle(f"{option_name} @{strike} Gamma (NAS={NAS_heatmap})", fontsize=14)
    else:
        fig.suptitle(f"{option_name}_{barrier_val} @{strike} Gamma (NAS={NAS_heatmap})", fontsize=14)

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

            grid = grid_data_hm.get("Gamma")
            S_vec = init_hm.get("S")
            time_vec = init_hm.get("time_grid")

            if grid is not None and S_vec is not None and time_vec is not None:
                processed_gamma_grid = (grid >= 0).astype(int)
                grids.append(processed_gamma_grid)
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
            ims[i] = plot_single_heatmap(axes[i], grids[i], S_vecs[i], time_vecs[i], title, cmap=cmap_gamma_sign, vmin=vmin, vmax=vmax)
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

    if im_for_colorbar: # Upewnij się, że obrazek do colorbara istnieje
        cbar_label_gamma = "Gamma sign"
        cbar = fig.colorbar(im_for_colorbar, ax=axes.ravel().tolist(), label=cbar_label_gamma, shrink=1)
        cbar.set_ticks([0, 1])
        cbar.ax.set_yticklabels([r'$\Gamma < 0$ -> $\sigma_-$', r'$\Gamma \geq 0$ -> $\sigma_+$'])

    # Adjust layout and save
    #plt.tight_layout(rect=[0, 0.03, 0.95, 0.95]) # Adjust rect to prevent title overlap and make space for colorbar
    fname = f"heatmap_1x3_fixed_vol_{option_code}_gamma_vols_{fixed_vols_for_1x3_heatmap[0]}_{fixed_vols_for_1x3_heatmap[1]}.png"
    filepath = fname
    try:
        plt.savefig(generated_path(filepath), bbox_inches='tight', dpi=300, format='png')
        plt.close(fig)
        print(f"  1x3 Heatmap saved to {filepath}")
        plot_files.append(filepath)
    except Exception as e:
        print(f"  Error saving 1x3 heatmap {filepath}: {e}")
        plt.close(fig)



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
        grid_fixed = processed_gamma_grid = (grid_data_fixed.get("Gamma") >= 0).astype(int)
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
        grid_unc = processed_gamma_grid = (grid_data_unc.get("Gamma") >= 0).astype(int)
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
        fname_comp = f"heatmap_comparison_{option_code}_gamma_vol_range_{vol_range_uncertain[0]}_{vol_range_uncertain[1]}.png"

        print(f"    Plotting comparative heatmap...")
        plot_file_comp = plot_comparative_heatmap(
            grid_fixed, S_vec_fixed, time_vec_fixed, title_fixed,
            grid_unc, S_vec_unc, time_vec_unc, title_unc,
            figure_title, fname_comp, cmap = cmap_gamma_sign, label=r"Gamma sign"
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
