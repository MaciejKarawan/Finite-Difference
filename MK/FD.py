from interpolation import interpolate_option_price
from pricing import *
import time


np.set_printoptions(
    precision=2,
    linewidth=200,
    suppress=True
)

############################ Explicit Finite Difference Method

global params
top_barrier = 3  # 3 x strike


def FD_wo_memory(NAS, s0=2000, top_barrier=3, vol=0.2, r=np.log(1 + 0.05), option_type='e_call', strike=2000, expiration=1):
    """
    Metoda różnic skończonych (Finite Difference) do wyceny opcji europejskich i amerykańskich.

    Parametry:
    - NAS: liczba kroków przestrzennych (Number of Asset Steps)
    - s0: początkowa cena aktywa
    - top_barrier: mnożnik górnej granicy siatki (jako wielokrotność ceny wykonania)
    - vol: zmienność
    - r: stopa procentowa
    - option_type: typ opcji ('e_call', 'e_put', 'a_call', 'a_put')
      gdzie 'e_' oznacza europejską, 'a_' oznacza amerykańską
    - strike: cena wykonania
    - expiration: czas do wygaśnięcia w latach

    Zwraca:
    - numerical_price: wyceniona wartość opcji
    - output: słownik z dodatkowymi informacjami
    - init: słownik z parametrami inicjalizacyjnymi
    """


    #### Inicjalizacja
    dS = top_barrier * strike / NAS                 # dS
    dt = 0.9 / (vol ** 2 * NAS ** 2)                # warunek na dt zapewniający stabilność
    NTS = int(expiration / dt) + 1                  # Number Time Steps
    dt = expiration / NTS                           # dla pewności całkowitości
    V = np.zeros(NAS)                               # wektor cen opcji (brak pamięci)
    S = np.cumsum(np.ones_like(V) * dS)[::-1]       # Wektor cen aktywa (malejąco)

    # Wektor czasów
    time_grid = np.linspace(0, expiration, NTS)

    print_it = False
    if print_it:
        print(f'Opcja: {option_type}, Strike: {strike}, dla ceny s0: {s0}')
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        print(f'Asset_vec: {S}')

    # Warunek końcowy (wypłata w terminie wygaśnięcia)
    current_time = expiration - 0 * dt

    # Ustalamy funkcję wypłaty w zależności od typu opcji
    if option_type in ['a_call', 'e_call']:
        V = np.maximum(S - strike, 0)
        payoff = lambda s: np.maximum(s - strike, 0)
    elif option_type in ['a_put', 'e_put']:
        V = np.maximum(strike - S, 0)
        payoff = lambda s: np.maximum(strike - s, 0)

    # Obliczenie współczynników dla schematu różnicowego
    a_k = 0.5 * vol ** 2 * S ** 2
    b_k = r * S
    c_k = -r

    v1 = dt / dS ** 2
    v2 = dt / dS

    A_k = v1 * a_k - 0.5 * v2 * b_k
    B_k = -2 * v1 * a_k + dt * c_k
    C_k = v1 * a_k + 0.5 * v2 * b_k

    start_time = time.perf_counter()

    # Główna pętla (wstecz w czasie)
    for t in range(1, NTS):
        current_time -= dt

        # Zapisujemy aktualne wartości
        V_i = V.copy()

        ### Przygotowanie wektorów dla schematu różnicowego
        # Dolny warunek brzegowy
        V_down = np.roll(V_i, -1)
        if option_type in ['e_put', 'a_put']:
            V_down[-1] = strike * np.exp(-r * (expiration - current_time))
        else:  # Call options
            V_down[-1] = 0

        # Górny warunek brzegowy
        V_up = np.roll(V_i, 1)
        if option_type in ['e_call', 'a_call']:
            V_up[0] = S[0] + dS - strike * np.exp(-r * (expiration - current_time))
        else:  # Put options
            V_up[0] = 0

        # Jawny schemat różnicowy
        V = A_k * V_down + (1 + B_k) * V_i + C_k * V_up

        # Dla opcji amerykańskiej sprawdzamy w każdym kroku, czy wcześniejsze wykonanie jest opłacalne
        if option_type.startswith('a_'):  # Opcje amerykańskie
            intrinsic = payoff(S)
            V = np.maximum(V, intrinsic)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call'
    else:
        theoretical_type = 'put'

    # Dla opcji amerykańskich typu call na aktywa bez dywidendy, wartość teoretyczna
    # jest taka sama jak dla opcji europejskiej

    if type(s0) is int or type(s0) is float:
        theoretical_price = option_price(s0, strike, expiration, r, vol, theoretical_type)
        numerical_price = interpolate_option_price(S, V, s0)
        diff = np.abs(numerical_price - theoretical_price)
        output = {
            'return': numerical_price,
            'theoretical': theoretical_price,
            'diff': diff,
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
        'time': execution_time
    }

    return numerical_price, output, init



def FD_with_memory(NAS, s0=2000, top_barrier=3, vol=0.2, r=np.log(1 + 0.05), option_type='e_call',
                   strike=2000, expiration=1):
    """
    Metoda różnic skończonych (Finite Difference) do wyceny opcji europejskich i amerykańskich
    z zapisem całej macierzy cen opcji w czasie.

    Parametry:
    - NAS: liczba kroków przestrzennych (Number of Asset Steps)
    - s0: początkowa cena aktywa
    - top_barrier: mnożnik górnej granicy siatki (jako wielokrotność ceny wykonania)
    - vol: zmienność
    - r: stopa procentowa
    - option_type: typ opcji ('e_call', 'e_put', 'a_call', 'a_put')
      gdzie 'e_' oznacza europejską, 'a_' oznacza amerykańską
    - strike: cena wykonania
    - expiration: czas do wygaśnięcia w latach

    Zwraca:
    - numerical_price: wyceniona wartość opcji
    - output: słownik z dodatkowymi informacjami
    - grid_data: słownik z macierzą cen opcji i innymi danymi siatki
    """

    #### Inicjalizacja
    dS = top_barrier * strike / NAS                       # dS
    dt = 0.9 / (vol ** 2 * NAS ** 2)                      # warunek na dt zapewniający stabilność
    NTS = int(expiration / dt) + 1                        # Number Time Steps
    dt = expiration / NTS                                 # dla pewności całkowitości
    Grid = np.zeros((NAS, NTS))                           # Inicjalizacja macierzy cen opcji (wymiary: NAS x NTS)
    Gamma_Grid = np.zeros((NAS, NTS))                 # Macierz gamma do śledzenia wartości gamma w każdym kroku czasowym
    Early_Exercise = np.zeros((NAS, NTS))             # Macierz do śledzenia wcześniejszych wykonań dla opcji amerykańskich
    S = np.cumsum(np.ones(NAS) * dS)[::-1]                # Wektor cen aktywa (ten sam dla każdego kroku czasowego)
    time_grid = np.linspace(0, expiration, NTS)      # Wektor czasów

    # Debug info
    print_it = False
    if print_it:
        print(f'Opcja: {option_type}, Strike: {strike}, dla ceny s0: {s0}')
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        print(f'Wymiary siatki: {Grid.shape}')

    # Ustawienie warunku końcowego (wypłata w terminie wygaśnięcia) - ostatnia kolumna macierzy
    if option_type in ['e_call', 'a_call']:
        Grid[:, -1] = np.maximum(S - strike, 0)
        payoff = lambda s: np.maximum(s - strike, 0)
    elif option_type in ['e_put', 'a_put']:
        Grid[:, -1] = np.maximum(strike - S, 0)
        payoff = lambda s: np.maximum(strike - s, 0)

    # Obliczenie współczynników dla schematu różnicowego
    v1 = dt / dS ** 2
    v2 = dt / dS

    a_k = 0.5 * vol ** 2 * S ** 2
    b_k = r * S
    c_k = -r

    A_k = v1 * a_k - 0.5 * v2 * b_k
    B_k = -2 * v1 * a_k + dt * c_k
    C_k = v1 * a_k + 0.5 * v2 * b_k

    start_time = time.perf_counter()

    # Główna pętla (wstecz w czasie)
    for t in range(1, NTS):
        index = (NTS - 1) - t
        current_time = expiration - t * dt

        # Pobieranie wartości z poprzedniego kroku czasowego
        V_i = Grid[:, index + 1]

        ### Przygotowanie wektorów dla schematu różnicowego
        # Dolny warunek brzegowy
        V_down = np.roll(V_i, -1)
        if option_type in ['e_put', 'a_put']:
            V_down[-1] = strike * np.exp(-r * (expiration - current_time))
        else:  # Call options
            V_down[-1] = 0

        # Górny warunek brzegowy
        V_up = np.roll(V_i, 1)
        if option_type in ['e_call', 'a_call']:
            V_up[0] = S[0] - strike * np.exp(-r * (expiration - current_time))
        else:  # Put options
            V_up[0] = 0

        Gamma_Grid[:, index] = (V_up - 2 * V_i + V_down) / (dS**2)      # Gamma
        Grid[:, index] = A_k * V_down + (1 + B_k) * V_i + C_k * V_up    # Schemat różnicowy

        # CHECK dla opcji amerykańskich
        if option_type.startswith('a_'):  # Opcje amerykańskie
            intrinsic = payoff(S)                                   # payoff
            early_exercise_mask = (intrinsic > Grid[:, index])      # Sprawdzamy, gdzie opłaca się wcześniejsze wykonanie
            Grid[:, index] = np.maximum(Grid[:, index], intrinsic)  # Aktualizujemy wartości opcji
            Early_Exercise[early_exercise_mask, index] = 1          # Oznaczamy miejsca wcześniejszego wykonania


    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call'
    else:
        theoretical_type = 'put'

    if type(s0) is int or type(s0) is float:
        theoretical_price = option_price(s0, strike, expiration, r, vol, theoretical_type)
        numerical_price = interpolate_option_price(S, Grid[:, 0], s0)  # Bierzemy pierwszy krok czasowy (czas=0)
        diff = np.abs(numerical_price - theoretical_price)

        # Wartość opcji w każdym kroku czasowym dla ceny s0
        price_evolution = []
        for t in range(NTS):
            interpolated_price = interpolate_option_price(S, Grid[:, t], s0)
            price_evolution.append(interpolated_price)

        output = {
            'return': numerical_price,
            'theoretical': theoretical_price,
            'diff': diff,
            "price_evolution": np.array(price_evolution),  # Ewolucja ceny opcji dla s0
        }

    else:
        numerical_price = Grid[:, 0]
        output = {
            'return': numerical_price
        }

    # Przygotowanie danych do zwrotu
    init = {
        "S": S,                         # Wektor cen aktywa
        "time_grid": time_grid,         # Wektor czasów
        "dS": dS,
        "dt": dt,
        "NAS": NAS,
        "NTS": NTS,
        'time': execution_time
    }
    Grid_data = {
        'Grid': Grid
    }

    return numerical_price, output, init, Grid_data