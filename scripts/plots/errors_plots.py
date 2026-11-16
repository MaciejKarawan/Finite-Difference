from finite_difference.paths import generated_path
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import seaborn as sns

# Ensure modules are in the same directory or Python path
try:
    from finite_difference import fd as FD
    from finite_difference import fd_ko as FD_KO
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
vol_fixed = 0.25      # Fixed volatility for non-barrier options
vol_range = [0.2, 0.2] # Volatility range for barrier options
barrier_call = 2400
barrier_put = 1900
NAS_range = [50, 100, 200, 400, 800] # Number of Asset Steps range for error plots
NAS_heatmap = 100      # NAS for heatmap generation
BENCHMARK_NAS = 2000   # NAS for calculating benchmark price when theoretical is unavailable



def run_error_analysis(option_type, barrier=None):
    """Runs FD methods for different NAS, collects errors and times."""
    results_wo = {"nas": [], "error": [], "time": []}
    results_with = {"nas": [], "error": [], "time": []}

    is_barrier = barrier is not None

    # --- Determine if theoretical price is available and should be used ---
    use_theoretical = False
    if option_type.startswith('e_'): # All European options have theoretical price
        use_theoretical = True
    elif option_type == 'a_call' and not is_barrier: # American call without barrier has theoretical price (same as European)
        use_theoretical = True
    # Otherwise (American Put, American Call/Put with barrier), use benchmark

    error_source_desc = "Theoretical Price" if use_theoretical else "High-NAS Benchmark"
    barrier_str = f"(Barrier: {barrier})" if is_barrier else ""
    print(f"--- Running Error Analysis for: {option_type} {barrier_str} (vs {error_source_desc}) ---")

    # --- Determine functions and common parameters --- #
    benchmark_price = np.nan # Initialize benchmark price

    if is_barrier:
        func_wo = FD_KO.FDB_wo_memory
        func_with = FD_KO.FDB_with_memory
        # Parameters for barrier options (always use vol_range)
        params_common = {"s0": s0, "vol": vol_range, "r": r, "option_type": option_type, "strike": strike, "expiration": expiration, "barrier": barrier}
        # Volatility for benchmark calculation (using FDB_wo_memory)
        bench_vol = vol_range
    else:
        func_wo = FD.FD_wo_memory
        func_with = FD.FD_with_memory
        # Parameters for non-barrier options (always use vol_fixed)
        params_common = {"s0": s0, "vol": vol_fixed, "r": r, "option_type": option_type, "strike": strike, "expiration": expiration}
        # Volatility for benchmark calculation (using FD_wo_memory)
        bench_vol = vol_fixed

    # --- Calculate Benchmark Price (only if needed) --- #
    if not use_theoretical:
        print(f"Calculating high-fidelity benchmark (NAS={BENCHMARK_NAS})...")
        bench_func = func_wo # Use the _wo_memory function for benchmark
        bench_params = params_common.copy()
        bench_params["vol"] = bench_vol # Ensure correct vol type/value for benchmark func

        try:
            # FDB_wo_memory returns 3 values, FD_wo_memory returns 3 values
            _, bench_output, _ = bench_func(NAS=BENCHMARK_NAS, **bench_params)
            benchmark_price = bench_output.get("return", np.nan)
            if np.isinf(benchmark_price) or np.isnan(benchmark_price) or np.abs(benchmark_price) > 1e10:
                print(f"  Warning: Benchmark price seems invalid ({benchmark_price}). Setting to NaN.")
                benchmark_price = np.nan
            else:
                print(f"  Benchmark Price: {benchmark_price:.4f}")
        except Exception as e:
            print(f"  Error calculating benchmark price: {e}. Setting to NaN.")
            benchmark_price = np.nan

    # --- Run Analysis for each NAS --- #
    for nas in NAS_range:
        print(f"  Running NAS = {nas}...")
        current_params = params_common.copy()
        # Ensure correct vol is passed for this NAS run (should match params_common logic)
        current_params["vol"] = vol_range if is_barrier else vol_fixed

        try:
            # Run without memory
            _, output_wo, init_wo = func_wo(NAS=nas, **current_params)
            time_wo = init_wo.get("time", np.nan)

            # Run with memory
            _, output_with, init_with, _ = func_with(NAS=nas, **current_params)
            time_with = init_with.get("time", np.nan)

            # --- Calculate Error --- #
            if use_theoretical:
                # Use 'diff' returned by the functions (absolute difference vs theoretical)
                error_wo = output_wo.get("diff", np.nan)
                error_with = output_with.get("diff", np.nan)
                # Check if theoretical price calculation failed (e.g., returned NaN in FD_KO)
                if np.isnan(error_wo) or np.isnan(error_with):
                    print(f"    Warning: Theoretical price/diff calculation might have failed for NAS={nas}.")
            else:
                # Use benchmark price
                if np.isnan(benchmark_price):
                    error_wo = np.nan # Cannot calculate error if benchmark is invalid
                    error_with = np.nan
                else:
                    price_wo = output_wo.get("return", np.nan)
                    price_with = output_with.get("return", np.nan)
                    error_wo = np.abs(price_wo - benchmark_price) if not np.isnan(price_wo) else np.nan
                    error_with = np.abs(price_with - benchmark_price) if not np.isnan(price_with) else np.nan

        except Exception as e:
            print(f"    Error during calculation for NAS={nas}: {e}")
            error_wo, time_wo = np.nan, np.nan
            error_with, time_with = np.nan, np.nan

        # --- Store results --- #
        # Handle potential NaN and zero errors for log plot
        results_wo["nas"].append(nas)
        results_wo["error"].append(error_wo if error_wo > 1e-15 else 1e-15)
        results_wo["time"].append(time_wo)

        results_with["nas"].append(nas)
        results_with["error"].append(error_with if error_with > 1e-15 else 1e-15)
        results_with["time"].append(time_with)

    return results_wo, results_with, use_theoretical

def plot_error_comparison(results_wo, results_with, use_theoretical, option_type_name, filename):
    """Plots the log-error vs NAS comparison, indicating the error source."""
    plt.figure(figsize=(10, 6))

    # --- Filter out NaN errors and times before plotting --- #
    valid_indices_wo = ~np.isnan(results_wo["error"]) & ~np.isnan(results_wo["time"])
    valid_indices_with = ~np.isnan(results_with["error"]) & ~np.isnan(results_with["time"])

    nas_wo = np.array(results_wo["nas"])[valid_indices_wo]
    error_wo = np.array(results_wo["error"])[valid_indices_wo]
    time_wo = np.array(results_wo["time"])[valid_indices_wo]

    nas_with = np.array(results_with["nas"])[valid_indices_with]
    error_with = np.array(results_with["error"])[valid_indices_with]
    time_with = np.array(results_with["time"])[valid_indices_with]

    # --- Plotting --- #
    if len(nas_wo) > 0:
        plt.plot(nas_wo, np.log10(error_wo), marker="o", linestyle="-", label="Without Memory")
        for i, nas in enumerate(nas_wo):
            plt.text(nas, np.log10(error_wo[i]), f" {time_wo[i]:.2f}s", fontsize=8, verticalalignment="bottom")
    else:
        print(f"Warning: No valid data points to plot for 'Without Memory' in {option_type_name}")

    if len(nas_with) > 0:
        plt.plot(nas_with, np.log10(error_with), marker="x", linestyle="--", label="With Memory")
        for i, nas in enumerate(nas_with):
            plt.text(nas, np.log10(error_with[i]), f" {time_with[i]:.2f}s", fontsize=8, verticalalignment="top")
    else:
        print(f"Warning: No valid data points to plot for 'With Memory' in {option_type_name}")

    # --- Labels and Title --- #
    plt.xlabel("Number of Asset Steps (NAS)")
    plt.ylabel("Log10(Absolute Error)")

    # Determine error source string based on the flag passed from run_error_analysis
    error_source = "Theoretical Price" if use_theoretical else "High-NAS Benchmark"
    plt.title(f"Error Convergence vs. NAS for {option_type_name}\n(Error vs {error_source}, Time in seconds)")

    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xticks(NAS_range) # Use original NAS range for ticks

    # --- Save Plot --- #
    filepath = filename # Save directly with the filename in the current directory
    try:
        plt.savefig(generated_path(filepath), bbox_inches='tight', dpi=300, format='png')
        plt.close()
        print(f"Plot saved to {filepath}")
    except Exception as e:
        print(f"Error saving plot {filepath}: {e}")
        plt.close() # Close plot even if saving failed
    return filepath



# == Task 1: Error Comparison Plots ==
plot_files = []
print("--- Generating Error Comparison Plots ---")

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

# Loop through all defined options
for code, (name, barrier_val) in option_definitions.items():
    # Extract base option type (e.g., 'e_call' from 'e_call_ko')
    base_option_type = code.split('_ko')[0]

    # Run analysis, returns results and whether theoretical price was used
    res_wo, res_with, use_theo = run_error_analysis(option_type=base_option_type, barrier=barrier_val)

    # Construct filename and full display name
    fname = f"error_plot_{code}.png"
    full_name = f"{name}{f' (Barrier={barrier_val})' if barrier_val else ''}"

    # Plot results, passing the use_theoretical flag
    plot_file = plot_error_comparison(res_wo, res_with, use_theo, full_name, fname)
    plot_files.append(plot_file)

print("\n--- Error Comparison Plots Generated ---")
error_plot_files = [f for f in plot_files if "error_plot" in os.path.basename(f)]
for f in error_plot_files:
    print(f"  {f}")