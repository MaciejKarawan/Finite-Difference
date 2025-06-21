#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from interpolation import interpolate_option_price
from pricing import *
import time
import numpy as np # Ensure numpy is imported

np.set_printoptions(
    precision=2,
    linewidth=200,
    suppress=True
)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


############################ Explicit Finite Difference Barrier Method

def FDB_wo_memory(NAS, s0=2000,
                  vol=[0.2, 0.3],  # Przedział zmienności [min, max]
                  r=np.log(1 + 0.05),
                  option_type='e_call',
                  strike=2000,
                  expiration=1,
                  barrier=2300,
                  # --- NEW Dividend Parameters ---
                  is_dividend=False,
                  dividend_time=0.5,
                  dividend_value=200
                  # --- End NEW Dividend Parameters ---
                  ):
    """
    Metoda różnic skończonych (Finite Difference) do wyceny opcji barierowych typu knock-out
    z uwzględnieniem niepewnej zmienności i opcjonalnej dyskretnej dywidendy.
    Minimalnie zmodyfikowana wersja.

    Parametry:
    - NAS: liczba kroków przestrzennych (Number of Asset Steps)
    - s0: początkowa cena aktywa
    - vol: przedział zmienności [min, max]
    - r: stopa procentowa
    - option_type: typ opcji (
      'e_call', 'e_put', 'a_call', 'a_put')
      gdzie 'e_' oznacza europejską, 'a_' oznacza amerykańską
    - strike: cena wykonania
    - expiration: czas do wygaśnięcia w latach
    - barrier: poziom bariery knock-out
    - is_dividend: bool, czy uwzględniać dywidendę
    - dividend_time: float, czas wypłaty dywidendy jako ułamek czasu do wygaśnięcia (0 < time < 1)
    - dividend_value: int lub float, wartość dywidendy (int = stała kwota, float = procent ceny aktywa)

    Zwraca:
    - numerical_price: wyceniona wartość opcji
    - output: słownik z dodatkowymi informacjami
    - init: słownik z parametrami inicjalizacyjnymi
    """

    barrier_type = 'up' if option_type in ['e_call', 'a_call'] else 'down'

    #### Inicjalizacja (Original Structure)
    if barrier_type == 'up':
        max_S = barrier
        min_S = 0
        dS = max_S / NAS
    elif barrier_type == 'down':
        min_S = barrier
        # Original max_S calculation
        max_S = 4 * strike
        dS = (max_S - min_S) / NAS

    # Original dt calculation
    if barrier_type == 'down':
        dt = 0.6 / (vol[1] ** 2 * NAS ** 2)
    else:
        dt = 0.9 / (vol[1] ** 2 * NAS ** 2)

    NTS = int(expiration / dt) + 1
    dt = expiration / NTS # Original adjustment
    S = np.linspace(max_S, min_S, NAS + 1) # Original S grid (descending)
    # Original time_grid definition (ascending, NTS points)
    # Note: This grid might not be directly used in the loop logic below
    # The loop uses `current_time = expiration - t * dt`
    time_grid_original = np.linspace(0, expiration, NTS)

    print_it = False
    if print_it:
        print(f'Opcja: {option_type}, Strike: {strike}, dla ceny s0: {s0} z barierą: {barrier} ({barrier_type})')
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        # print(f'Asset_vec: {S}')

    # Ustalamy funkcję wypłaty (Original)
    if option_type in ['a_call', 'e_call']:
        payoff = lambda s: np.maximum(s - strike, 0)
    elif option_type in ['a_put', 'e_put']:
        payoff = lambda s: np.maximum(strike - s, 0)
    else:
        raise ValueError("Nieznany typ opcji")

    # Warunek końcowy z uwzględnieniem bariery (Original)
    if barrier_type == 'up':
        barrier_mask = S < barrier
    elif barrier_type == 'down':
        barrier_mask = S > barrier
    else:
        raise ValueError("Nieznany typ bariery")

    V = np.zeros_like(S)
    V[barrier_mask] = payoff(S[barrier_mask])

    start_time = time.perf_counter()

    # --- NEW: Find dividend time step index --- #
    # We need the index `t` in the loop `for t in range(1, NTS):`
    # such that `current_time = expiration - t * dt` is just *before* the dividend payment time.
    # Dividend payment time = expiration * dividend_time (time from start)
    # Time just before dividend = expiration * dividend_time
    # We are stepping back from expiration. The step `t` corresponds to time `expiration - t*dt`.
    # We want the step `t` where `expiration - t*dt` is the first time point <= `expiration * dividend_time`.
    # `expiration - t*dt <= expiration * dividend_time`
    # `expiration * (1 - dividend_time) <= t * dt`
    # `t >= expiration * (1 - dividend_time) / dt`
    dividend_step_t = -1
    if is_dividend and 0 < dividend_time < 1:
        target_t = expiration * (1 - dividend_time) / dt
        dividend_step_t = int(np.ceil(target_t))
        # Ensure it's within the loop bounds (1 to NTS-1)
        if dividend_step_t <= 0 or dividend_step_t >= NTS:
            print(f"Warning: Dividend time {dividend_time*expiration:.4f} maps to step {dividend_step_t}, outside loop range [1, {NTS-1}]. Disabling dividend.")
            dividend_step_t = -1
        else:
             print(f"Dividend adjustment will be applied at loop step t = {dividend_step_t}, corresponding to time ~ {expiration - dividend_step_t * dt:.4f}")
    # --- End NEW --- #

    # Original constant coefficients calculation
    b_k = r * S
    c_k = -r
    v1 = dt / dS ** 2
    v2 = dt / dS

    # Original Main loop (backward in time)
    for t in range(1, NTS):
        current_time = expiration - t * dt
        V_prev_step = V.copy() # Values from the previous step (later in time)

        # --- NEW: Dividend Adjustment --- #
        # Check if the *current* step `t` is the one where adjustment should happen
        if is_dividend and t == dividend_step_t:
            V_after_dividend = V_prev_step
            V_before_dividend = np.zeros_like(V_after_dividend)

            # Calculate S_prime = S - D(S)
            if isinstance(dividend_value, int) or isinstance(dividend_value, float) and dividend_value.is_integer():
                D_val = float(dividend_value)
                S_prime = S - D_val
            elif isinstance(dividend_value, float):
                D_val_pct = dividend_value
                S_prime = S * (1.0 - D_val_pct)
            else:
                print("Warning: Invalid dividend_value type. Assuming zero dividend.")
                S_prime = S

            # Interpolate V_after_dividend at S_prime to get V_before_dividend
            # np.interp requires x-coordinates to be increasing, S is decreasing.
            V_before_dividend = np.interp(S_prime, S[::-1], V_after_dividend[::-1])

            # Apply barrier condition again after interpolation
            if barrier_type == 'up':
                V_before_dividend[S_prime >= barrier] = 0
            elif barrier_type == 'down':
                V_before_dividend[S_prime <= barrier] = 0
            # Ensure values outside original barrier mask are also zero
            V_before_dividend[~barrier_mask] = 0

            # Use the adjusted values as the input for the FDM step
            V_i = V_before_dividend
        else:
            # Standard case: use values from the previous time step
            V_i = V_prev_step
        # --- End NEW --- #

        # Original Boundary conditions using roll (applied to V_i)
        V_up = np.roll(V_i, 1)
        V_down = np.roll(V_i, -1)
        V_up[0] = 0
        V_down[-1] = 0

        # Original Gamma and volatility calculation (using V_i)
        gamma = (V_up - 2 * V_i + V_down) / (dS ** 2)
        # Handle potential division by zero or NaN at boundaries if needed (simple extrapolation)
        gamma[np.isnan(gamma)] = 0 # Replace NaN if any
        gamma[0] = gamma[1]
        gamma[-1] = gamma[-2]
        sigma = np.where(gamma > 0, vol[1], vol[0])

        # Original FDM coefficients calculation
        a_k = 0.5 * sigma ** 2 * S ** 2
        A_k = v1 * a_k - 0.5 * v2 * b_k
        B_k = -2 * v1 * a_k + dt * c_k
        C_k = v1 * a_k + 0.5 * v2 * b_k

        # Original Explicit step calculation
        # V(t) = A*V(t+1)[i-1] + (1+B)*V(t+1)[i] + C*V(t+1)[i+1]
        # Uses V_down, V_i, V_up calculated from the potentially adjusted V_i
        V_next = A_k * V_down + (1 + B_k) * V_i + C_k * V_up

        # Original Barrier cutoffs
        # Apply barrier based on S grid, not S_prime
        V_next[~barrier_mask] = 0 # Ensure values outside barrier are zero
        # Redundant checks from original code (already covered by barrier_mask)
        # if option_type in ['e_call', 'a_call']:
        #     V_next[0] = 0
        # elif option_type in ['e_put', 'a_put']:
        #     V_next[-1] = 0

        # Original American exercise check
        if option_type.startswith('a_'):
            intrinsic_value = payoff(S)
            # Early exercise only possible if not knocked out
            V_next = np.where(barrier_mask, np.maximum(V_next, intrinsic_value), 0)

        V = V_next # Update V for the next iteration

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Original calculation of theoretical price and output
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call_up_and_out'
    else:
        theoretical_type = 'put_down_and_out'

    numerical_price = 0
    theoretical_price = None # Set to None as discrete dividend complicates theoretical price
    output = {}

    if isinstance(s0, (int, float)):
        if (barrier_type == 'up' and s0 >= barrier) or (barrier_type == 'down' and s0 <= barrier):
            numerical_price = 0
            # Attempt to calculate theoretical even if numerical is 0
            try:
                # Need gamma at t=0, which wasn't stored. Approximate using final gamma.
                _, id_s0 = find_nearest(S, s0)
                # Use gamma from the step before last (t=NTS-1), index 1 if NTS=2
                # This gamma is not available here. Cannot calculate theoretical accurately.
                # sigma_0 = vol[0] if gamma[id_s0] > 0 else vol[1]
                # theoretical_price = option_barrier_price(s0, strike, barrier, expiration, r, sigma_0, theoretical_type)
                theoretical_price = None # Cannot calculate easily
            except Exception:
                theoretical_price = None
        else:
            # Interpolate the final result V (which corresponds to time=0)
            numerical_price = np.interp(s0, S[::-1], V[::-1])
            try:
                # Cannot calculate theoretical easily
                theoretical_price = None
            except Exception:
                theoretical_price = None

        output = {
            'return': numerical_price,
            'theoretical': theoretical_price,
            'diff': np.abs(numerical_price - theoretical_price) if theoretical_price is not None else None,
        }
    else: # If s0 is not a single value, return the whole vector at t=0
        numerical_price = V
        output = {
            'return': numerical_price
        }

    init = {
        "dS": dS,
        "NAS": NAS,
        "NTS": NTS,
        "dt": dt,
        "time_grid": time_grid_original, # Return original time grid
        "S": S,
        "barrier": barrier,
        "barrier_type": barrier_type,
        "vol_range": vol,
        "time": execution_time,
        # --- NEW --- #
        "is_dividend": is_dividend,
        "dividend_time": dividend_time,
        "dividend_value": dividend_value
        # --- End NEW --- #
    }

    return numerical_price, output, init


def FDB_with_memory(NAS, s0=2000,
                    vol=[0.2, 0.3],  # Przedział zmienności [min, max]
                    r=np.log(1 + 0.05),
                    option_type='e_call',
                    strike=2000,
                    expiration=1,
                    barrier=2300,
                    # --- NEW Dividend Parameters ---
                    is_dividend=False,
                    dividend_time=0.5,
                    dividend_value=200
                    # --- End NEW Dividend Parameters ---
                    ):
    """
    Metoda różnic skończonych (Finite Difference) do wyceny opcji barierowych typu knock-out
    z niepewną zmiennością, zapisem całej macierzy cen opcji i opcjonalną dyskretną dywidendą.
    Minimalnie zmodyfikowana wersja.

    Parametry:
    - NAS: liczba kroków przestrzennych (Number of Asset Steps)
    - s0: początkowa cena aktywa
    - vol: przedział zmienności [min, max]
    - r: stopa procentowa
    - option_type: typ opcji (
      'e_call', 'e_put', 'a_call', 'a_put')
    - strike: cena wykonania
    - expiration: czas do wygaśnięcia w latach
    - barrier: poziom bariery knock-out
    - is_dividend: bool, czy uwzględniać dywidendę
    - dividend_time: float, czas wypłaty dywidendy jako ułamek czasu do wygaśnięcia (0 < time < 1)
    - dividend_value: int lub float, wartość dywidendy (int = stała kwota, float = procent ceny aktywa)

    Zwraca:
    - numerical_price: wyceniona wartość opcji
    - output: słownik z dodatkowymi informacjami
    - init: słownik z parametrami inicjalizacyjnymi
    - grid_data: słownik z macierzą cen opcji i innymi danymi siatki
    """

    barrier_type = 'up' if option_type in ['e_call', 'a_call'] else 'down'

    # Inicjalizacja (Original Structure)
    if barrier_type == 'up':
        max_S = barrier
        min_S = 0
        dS = max_S / NAS
    else:
        min_S = barrier
        max_S = 4 * strike # Original max_S
        dS = (max_S - min_S) / NAS

    # Original dt calculation
    if barrier_type == 'down':
        dt = 0.6 / (vol[1] ** 2 * NAS ** 2)
    else:
        dt = 0.9 / (vol[1] ** 2 * NAS ** 2)

    NTS = int(expiration / dt) + 1
    dt = expiration / NTS # Original adjustment

    # Original Grid dimensions and setup
    Grid = np.zeros((NAS + 1, NTS))
    Gamma_Grid = np.zeros((NAS + 1, NTS))
    Early_Exercise = np.zeros((NAS + 1, NTS))
    S = np.linspace(max_S, min_S, NAS + 1) # Original S grid (descending)
    time_grid_original = np.linspace(0, expiration, NTS) # Original time grid (ascending)

    # Debug info (Original)
    print_it = False
    if print_it:
        print(f'Opcja: {option_type}, Strike: {strike}, dla ceny s0: {s0} z barierą: {barrier} ({barrier_type})')
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        print(f'Wymiary siatki: {Grid.shape}')

    # Ustalamy funkcję wypłaty (Original)
    if option_type in ['a_call', 'e_call']:
        payoff = lambda s: np.maximum(s - strike, 0)
    elif option_type in ['a_put', 'e_put']:
        payoff = lambda s: np.maximum(strike - s, 0)
    else:
        raise ValueError("Nieznany typ opcji")

    # Warunek końcowy z uwzględnieniem bariery (Original)
    if barrier_type == 'up':
        barrier_mask = S < barrier
    else:  # barrier_type == 'down'
        barrier_mask = S > barrier

    # Wypełniamy ostatnią kolumnę macierzy (Original: index -1)
    Grid[:, -1] = np.zeros_like(S)
    Grid[barrier_mask, -1] = payoff(S[barrier_mask])

    # --- NEW: Find dividend time step index --- #
    # The loop uses `index = (NTS - 1) - t` where `t` goes from 1 to NTS-1.
    # `index` goes from NTS-2 down to 0.
    # `index` corresponds to time `time_grid_original[index]` or `index * dt`.
    # We want the step `index` just *before* the dividend payment time.
    # Dividend payment time = expiration * dividend_time
    # We want the largest index `idx` such that `idx * dt < expiration * dividend_time`.
    dividend_step_index = -1
    if is_dividend and 0 < dividend_time < 1:
        target_time = expiration * dividend_time
        # Find the index corresponding to the time step *after* the dividend payment
        target_idx_after = int(np.ceil(target_time / dt))
        # The adjustment happens when calculating the step *before* this time
        dividend_step_index = target_idx_after - 1 # This is the index we calculate

    '''    if dividend_step_index < 0 or dividend_step_index >= NTS -1:
             print(f"Warning: Dividend time {target_time:.4f} maps to index {dividend_step_index}, outside calculation range [0, {NTS-2}]. Disabling dividend.")
             dividend_step_index = -1
        else:
             print(f"Dividend adjustment will be applied when calculating index = {dividend_step_index}, corresponding to time ~ {dividend_step_index * dt:.4f}")
    # --- End NEW --- #'''

    # Original constant coefficients calculation
    b_k = r * S
    c_k = -r
    v1 = dt / dS ** 2
    v2 = dt / dS

    start_time = time.perf_counter()

    # Główna pętla (wstecz w czasie) - Original Structure
    for t in range(1, NTS):
        index = (NTS - 1) - t                   # Index of the column we are calculating (NTS-2 down to 0)
        V_prev_step = Grid[:, index + 1]        # Values from the previous step (later in time)

        # --- NEW: Dividend Adjustment --- #
        # Check if the *current* index `index` is the one where adjustment should happen
        if is_dividend and index == dividend_step_index:
            V_after_dividend = V_prev_step
            V_before_dividend = np.zeros_like(V_after_dividend)

            if isinstance(dividend_value, int) or isinstance(dividend_value, float) and dividend_value.is_integer():
                D_val = float(dividend_value)
                S_prime = S - D_val
            elif isinstance(dividend_value, float):
                D_val_pct = dividend_value
                S_prime = S * (1.0 - D_val_pct)
            else:
                print("Warning: Invalid dividend_value type. Assuming zero dividend.")
                S_prime = S

            V_before_dividend = np.interp(S_prime, S[::-1], V_after_dividend[::-1])

            if barrier_type == 'up':
                V_before_dividend[S_prime >= barrier] = 0
            elif barrier_type == 'down':
                V_before_dividend[S_prime <= barrier] = 0
            V_before_dividend[~barrier_mask] = 0

            V_i = V_before_dividend
        else:
            V_i = V_prev_step
        # --- End NEW --- #

        # Original Boundary conditions using roll (applied to V_i)
        V_up = np.roll(V_i, 1)
        V_down = np.roll(V_i, -1)
        V_up[0] = 0
        V_down[-1] = 0

        # Original Gamma and volatility calculation (using V_i)
        gamma = (V_up - 2 * V_i + V_down) / (dS ** 2)
        # Handle potential division by zero or NaN at boundaries
        gamma[np.isnan(gamma)] = 0
        gamma[0] = gamma[1]
        gamma[-1] = gamma[-2]
        # Store Gamma (Original - check boundary handling)
        Gamma_Grid[:, index] = gamma
        # Original boundary extrapolation for Gamma_Grid
        Gamma_Grid[0, index] = 2*Gamma_Grid[1, index] - Gamma_Grid[2, index]
        Gamma_Grid[-1, index] = 2*Gamma_Grid[-2, index] - Gamma_Grid[-3, index]

        sigma = np.where(gamma > 0, vol[1], vol[0])

        # Original FDM coefficients calculation
        a_k = 0.5 * sigma ** 2 * S ** 2
        A_k = v1 * a_k - 0.5 * v2 * b_k
        B_k = -2 * v1 * a_k + dt * c_k
        C_k = v1 * a_k + 0.5 * v2 * b_k

        # Original Explicit step calculation
        V_next = A_k * V_down + (1 + B_k) * V_i + C_k * V_up

        # Original Barrier cutoffs
        V_next[~barrier_mask] = 0
        # Redundant checks from original code
        # if option_type in ['e_call', 'a_call']:
        #     V_next[0] = 0
        # elif option_type in ['e_put', 'a_put']:
        #     V_next[-1] = 0

        # Store result in Grid (Original)
        Grid[:, index] = V_next

        # Original American exercise check (applied after storing in Grid)
        if option_type.startswith('a_'):
            intrinsic_value = payoff(S)
            early_exercise_mask = (intrinsic_value > Grid[:, index]) & barrier_mask
            Grid[early_exercise_mask, index] = intrinsic_value[early_exercise_mask]
            Early_Exercise[early_exercise_mask, index] = 1

    # Czas end (Original)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Original calculation of gamma at t=0
    # Need values at index 1 (t=dt) to calculate gamma at index 0 (t=0)
    if NTS > 1:
        V_i_t0 = Grid[:, 1] # Values at t=dt
        V_up_t0 = np.roll(V_i_t0, 1)
        V_down_t0 = np.roll(V_i_t0, -1)
        V_up_t0[0] = 0
        V_down_t0[-1] = 0
        Gamma_Grid[:, 0] = (V_up_t0 - 2 * V_i_t0 + V_down_t0) / (dS ** 2)
        Gamma_Grid[0, 0] = Gamma_Grid[1, 0]
        Gamma_Grid[-1, 0] = Gamma_Grid[-2, 0]
    else: # Handle case NTS=1
         Gamma_Grid[:, 0] = 0 # Or some other default

    # Original calculation of theoretical price and output
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call_up_and_out'
    else:
        theoretical_type = 'put_down_and_out'

    numerical_price = 0
    theoretical_price = None # Set to None
    price_evolution = None
    output = {}

    if isinstance(s0, (int, float)):
        # Use gamma at t=0 (index 0) for theoretical calculation if possible
        try:
            _, id_s0 = find_nearest(S, s0)
            # sigma_0 = vol[0] if Gamma_Grid[id_s0, 0] > 0 else vol[1]
            theoretical_price = option_barrier_price(s0 - dividend_value*np.exp(-r * dividend_time), strike, barrier, expiration, r, vol[0], theoretical_type)
        except Exception:
            theoretical_price = None

        if (barrier_type == 'up' and s0 >= barrier) or (barrier_type == 'down' and s0 <= barrier):
            numerical_price = 0
            price_evolution = np.zeros(NTS)
        else:
            # Interpolate the final result V at t=0 (index 0 in Grid)
            numerical_price = np.interp(s0, S[::-1], Grid[::-1, 0])
            # Interpolate price evolution over time (indices 0 to NTS-1)
            price_evolution = np.array([np.interp(s0, S[::-1], Grid[::-1, i]) for i in range(NTS)])

        output = {
            'return': numerical_price,
            'theoretical': theoretical_price,
            'diff': np.abs(numerical_price - theoretical_price) if theoretical_price is not None else None,
            "price_evolution": price_evolution, # Price evolution from t=0 to T
        }
    else:
        numerical_price = Grid[:, 0] # Return the whole vector at time=0
        output = {
            'return': numerical_price
        }

    init = {
        "S": S,
        "time_grid": time_grid_original, # Return original time grid
        "dS": dS,
        "dt": dt,
        "NAS": NAS,
        "NTS": NTS,
        "barrier": barrier,
        "barrier_type": barrier_type,
        "vol_range": vol,
        "time": execution_time,
        # --- NEW --- #
        "is_dividend": is_dividend,
        "dividend_time": dividend_time,
        "dividend_value": dividend_value
        # --- End NEW --- #
    }
    Grid_data = {
        'Grid': Grid,
        'Gamma': Gamma_Grid,
        # 'Sigma': sigma, # Sigma is calculated per step, not stored globally in original
        'Early_Exercise': Early_Exercise
    }

    return numerical_price, output, init, Grid_data

