from finite_difference.interpolation import interpolate_option_price
from finite_difference.pricing import *
import time

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
                  ):
    """
    Metoda różnic skończonych (Finite Difference) do wyceny opcji barierowych typu knock-out
    z uwzględnieniem niepewnej zmienności.

    Parametry:
    - NAS: liczba kroków przestrzennych (Number of Asset Steps)
    - s0: początkowa cena aktywa
    - vol: przedział zmienności [min, max]
    - r: stopa procentowa
    - option_type: typ opcji ('e_call', 'e_put', 'a_call', 'a_put')
      gdzie 'e_' oznacza europejską, 'a_' oznacza amerykańską
    - strike: cena wykonania
    - expiration: czas do wygaśnięcia w latach
    - barrier: poziom bariery knock-out

    Zwraca:
    - numerical_price: wyceniona wartość opcji
    - output: słownik z dodatkowymi informacjami
    - init: słownik z parametrami inicjalizacyjnymi
    """

    barrier_type = 'up' if option_type in ['e_call', 'a_call'] else 'down'

    #### Inicjalizacja
    if barrier_type == 'up':
        max_S = barrier
        min_S = 0
        dS = max_S / NAS
    elif barrier_type == 'down':
        min_S = barrier
        max_S = 4 * strike
        dS = (max_S - min_S) / NAS

    if barrier_type == 'down':
        dt = 0.6 / (vol[1] ** 2 * NAS ** 2)         # Warunek na dt zapewniający stabilność dla maksymalnej zmienności
    else:
        dt = 0.9 / (vol[1] ** 2 * NAS ** 2)

    NTS = int(expiration / dt) + 1                          # Number Time Steps
    dt = expiration / NTS                                   # dla pewności całkowitości
    S = np.linspace(max_S, min_S, NAS + 1)
    time_grid = np.linspace(0, expiration, NTS)


    print_it = False
    if print_it:
        print(f'Opcja: {option_type}, Strike: {strike}, dla ceny s0: {s0} z barierą: {barrier} ({barrier_type})')
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        print(f'Asset_vec: {S}')

    # Ustalamy funkcję wypłaty w zależności od typu opcji
    if option_type in ['a_call', 'e_call']:
        payoff = lambda s: np.maximum(s - strike, 0)
    elif option_type in ['a_put', 'e_put']:
        payoff = lambda s: np.maximum(strike - s, 0)

    # Warunek końcowy z uwzględnieniem bariery
    if barrier_type == 'up':
        barrier_mask = S < barrier
    elif barrier_type == 'down':
        barrier_mask = S > barrier

    V = np.zeros_like(S)
    V[barrier_mask] = payoff(S[barrier_mask])

    start_time = time.perf_counter()

    # część stałych współczynników schematu różnicowego
    b_k = r * S
    c_k = -r
    v1 = dt / dS ** 2
    v2 = dt / dS

    for t in range(1, NTS):
        current_time = expiration - t * dt
        V_i = V.copy()

        ### Warunki brzegowe
        V_up = np.roll(V_i, 1)
        V_down = np.roll(V_i, -1)
        V_up[0] = 0
        V_down[-1] = 0

        ### Gamma i volatility
        gamma = (V_up - 2 * V_i + V_down) / (dS ** 2)
        sigma = np.where(gamma > 0, vol[1], vol[0])

        ### Obliczenie współczynników dla schematu różnicowego
        a_k = 0.5 * sigma ** 2 * S ** 2
        A_k = v1 * a_k - 0.5 * v2 * b_k
        B_k = -2 * v1 * a_k + dt * c_k
        C_k = v1 * a_k + 0.5 * v2 * b_k

        V = A_k * V_down + (1 + B_k) * V_i + C_k * V_up         # Schemat różnicowy dla wewnętrznych punktów

        # Ucięcia z powodu barier
        if option_type in ['e_call', 'a_call']:
            V[0] = 0
        elif option_type in ['e_put', 'a_put']:
            V[-1] = 0

        ### Dla opcji amerykańskiej sprawdzamy wcześniejsze wykonanie
        if option_type.startswith('a_'):
            intrinsic_value = payoff(S)
            V = np.where(barrier_mask, np.maximum(V, intrinsic_value), 0)

    end_time = time.perf_counter()
    execution_time = end_time - start_time



    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call_up_and_out'
    else:
        theoretical_type = 'put_down_and_out'

    if type(s0) is int or type(s0) is float:

        _, id = find_nearest(V, s0)
        sigma_0 = vol[0] if gamma[id] > 0 else vol[1]
        theoretical_price = option_barrier_price(s0, strike, barrier, expiration, r, sigma_0, theoretical_type)

        if (barrier_type == 'up' and s0 >= barrier) or (barrier_type == 'down' and s0 <= barrier):
            numerical_price = 0
            output = {
                'return': numerical_price,
                'theoretical': theoretical_price,
                'diff': np.abs(numerical_price - theoretical_price),
            }

        else:
            numerical_price = interpolate_option_price(S, V, s0)
            output = {
                'return': numerical_price,
                'theoretical': theoretical_price,
                'diff': np.abs(numerical_price - theoretical_price),
            }
    else:
        numerical_price = V
        output = {
            'return': numerical_price
        }

    init = {
        "dS": dS,
        "NAS": NAS,
        "NTS": NTS,
        "dt": dt,
        "time_grid": time_grid,
        "S": S,
        "barrier": barrier,
        "barrier_type": barrier_type,
        "vol_range": vol,
        "time": execution_time
    }

    return numerical_price, output, init



def FDB_with_memory(NAS, s0=2000,
                    vol=[0.2, 0.3],  # Przedział zmienności [min, max]
                    r=np.log(1 + 0.05),
                    option_type='e_call',
                    strike=2000,
                    expiration=1,
                    barrier=2300):
    """
    Metoda różnic skończonych (Finite Difference) do wyceny opcji barierowych typu knock-out
    z niepewną zmiennością i zapisem całej macierzy cen opcji w czasie.

    Parametry:
    - NAS: liczba kroków przestrzennych (Number of Asset Steps)
    - s0: początkowa cena aktywa
    - vol: przedział zmienności [min, max]
    - r: stopa procentowa
    - option_type: typ opcji ('e_call', 'e_put', 'a_call', 'a_put')
    - strike: cena wykonania
    - expiration: czas do wygaśnięcia w latach
    - barrier: poziom bariery knock-out
    - barrier_type: typ bariery ('up' dla górnej, 'down' dla dolnej)

    Zwraca:
    - numerical_price: wyceniona wartość opcji
    - output: słownik z dodatkowymi informacjami
    - init: słownik z parametrami inicjalizacyjnymi
    - grid_data: słownik z macierzą cen opcji i innymi danymi siatki
    """

    barrier_type = 'up' if option_type in ['e_call', 'a_call'] else 'down'

    # Inicjalizacja
    if barrier_type == 'up':
        max_S = barrier
        min_S = 0
        dS = max_S / NAS
    else:
        min_S = barrier
        max_S = 4 * strike
        dS = (max_S - min_S) / NAS

    if barrier_type == 'down':
        dt = 0.6 / (vol[1] ** 2 * NAS ** 2)                     # Warunek na dt zapewniający stabilność dla maksymalnej zmienności
    else:
        dt = 0.9 / (vol[1] ** 2 * NAS ** 2)

    NTS = int(expiration / dt) + 1                              # Number Time Steps
    dt = expiration / NTS                                       # dla pewności całkowitości
    Grid = np.zeros((NAS + 1, NTS))                             # Inicjalizacja macierzy cen opcji (wymiary: (NAS+1) x NTS)
    Gamma_Grid = np.zeros((NAS + 1, NTS))                       # macierz gamma do śledzenia wartości gamma w każdym kroku czasowym
    Early_Exercise = np.zeros((NAS + 1, NTS))                   # Dodane: macierz do śledzenia wcześniejszych wykonań dla opcji amerykańskich
    S = np.linspace(max_S, min_S, NAS + 1)                      # Tworzymy siatkę cen aktywa - od najwyższej do najniższej
    time_grid = np.linspace(0, expiration, NTS)            # Wektor czasów

    # Debug info
    print_it = False
    if print_it:
        print(f'Opcja: {option_type}, Strike: {strike}, dla ceny s0: {s0} z barierą: {barrier} ({barrier_type})')
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        print(f'Wymiary siatki: {Grid.shape}')

    # Ustalamy funkcję wypłaty w zależności od typu opcji
    if option_type in ['a_call', 'e_call']:
        payoff = lambda s: np.maximum(s - strike, 0)
    elif option_type in ['a_put', 'e_put']:
        payoff = lambda s: np.maximum(strike - s, 0)

    # Warunek końcowy z uwzględnieniem bariery
    if barrier_type == 'up':
        barrier_mask = S < barrier
    else:  # barrier_type == 'down'
        barrier_mask = S > barrier

    # Wypełniamy ostatnią kolumnę macierzy
    Grid[:, -1] = np.zeros_like(S)
    Grid[barrier_mask, -1] = payoff(S[barrier_mask])

    # Przygotowanie wektorów dla obliczeń
    sigma = np.zeros_like(S)

    # część stałych współczynników schematu różnicowego
    b_k = r * S
    c_k = -r
    v1 = dt / dS ** 2
    v2 = dt / dS

    start_time = time.perf_counter()

    # Główna pętla (wstecz w czasie)
    for t in range(1, NTS):
        index = (NTS - 1) - t                   # Indeks kolumny, którą wypełniamy (idziemy od końca)
        current_time = expiration - t * dt
        V_i = Grid[:, index + 1]                # Pobieranie wartości z poprzedniego kroku czasowego

        # Warunki brzegowe
        V_up = np.roll(V_i, 1)
        V_down = np.roll(V_i, -1)
        V_up[0] = 0
        V_down[-1] = 0

        # Gamma i volatility
        gamma = (V_up - 2 * V_i + V_down) / (dS ** 2)
        Gamma_Grid[:, index] = gamma
        Gamma_Grid[0, index] = 2*Gamma_Grid[1, index] - Gamma_Grid[2, index]
        Gamma_Grid[-1, index] = 2*Gamma_Grid[-2, index] - Gamma_Grid[-3, index]

        sigma = np.where(gamma > 0, vol[1], vol[0])

        ### Obliczenie współczynników dla schematu różnicowego
        a_k = 0.5 * sigma ** 2 * S ** 2
        A_k = v1 * a_k - 0.5 * v2 * b_k
        B_k = -2 * v1 * a_k + dt * c_k
        C_k = v1 * a_k + 0.5 * v2 * b_k

        V = A_k * V_down + (1 + B_k) * V_i + C_k * V_up         # Schemat różnicowy dla wewnętrznych punktów

        # Ucięcia z powodu barier
        if option_type in ['e_call', 'a_call']:
            V[0] = 0
        elif option_type in ['e_put', 'a_put']:
            V[-1] = 0

        Grid[:, index] = V

        # Dla opcji amerykańskiej sprawdzamy wcześniejsze wykonanie
        if option_type.startswith('a_'):  # Opcje amerykańskie
            intrinsic_value = payoff(S)
            early_exercise_mask = (intrinsic_value > Grid[:, index]) & barrier_mask
            Grid[early_exercise_mask, index] = intrinsic_value[early_exercise_mask]
            Early_Exercise[early_exercise_mask, index] = 1

    # Czas end
    end_time = time.perf_counter()
    execution_time = end_time - start_time


    # Obliczamy gammę dla pierwszego kroku czasowego (t=0)
    V_i = Grid[:, 1]
    V_up = np.roll(V_i, 1)
    V_down = np.roll(V_i, -1)
    V_up[0] = 0
    V_down[-1] = 0
    Gamma_Grid[:, 0] = (V_up - 2 * V_i + V_down) / (dS ** 2)

    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call_up_and_out'
    else:
        theoretical_type = 'put_down_and_out'

    if type(s0) is int or type(s0) is float:

        _, id = find_nearest(V, s0)
        sigma_0 = vol[0] if Gamma_Grid[id, 0] > 0 else vol[1]
        theoretical_price = option_barrier_price(s0, strike, barrier, expiration, r, sigma_0, theoretical_type)

        if (barrier_type == 'up' and s0 >= barrier) or (barrier_type == 'down' and s0 <= barrier):
            numerical_price = 0
            price_evolution = np.zeros(NTS)
            output = {
                'return': numerical_price,
                'theoretical': theoretical_price,
                'diff': np.abs(numerical_price - theoretical_price),
                "price_evolution": np.array(price_evolution),
            }

        else:
            numerical_price = interpolate_option_price(S, V, s0)
            price_evolution = []
            for t in range(NTS):
                interpolated_price = interpolate_option_price(S, Grid[:, t], s0)
                price_evolution.append(interpolated_price)

            output = {
                'return': numerical_price,
                'theoretical': theoretical_price,
                'diff': np.abs(numerical_price - theoretical_price),
                "price_evolution": np.array(price_evolution),
            }

    else:
        numerical_price = Grid[:, 0]
        output = {
            'return': numerical_price
        }

    init = {
        "S": S,  # Wektor cen aktywa
        "time_grid": time_grid,  # Wektor czasów
        "dS": dS,
        "dt": dt,
        "NAS": NAS,
        "NTS": NTS,
        "barrier": barrier,
        "barrier_type": barrier_type,
        "vol_range": vol,
        "time": execution_time
    }
    Grid_data = {
        'Grid': Grid,
        'Gamma': Gamma_Grid,  # Pełna macierz gamma
        'Sigma': sigma,  # Wektor zmienności
        'Early_Exercise': Early_Exercise  # Macierz wcześniejszych wykonań
    }

    return numerical_price, output, init, Grid_data