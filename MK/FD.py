from interpolation import interpolate_option_price
from pricing import *

np.set_printoptions(
    precision=2,
    linewidth=200,
    suppress=True
)

############################ Explicit Finite Difference Method

### Args
vol = 0.20
r = np.log(1 + 0.05)
option_type = 'e_call'
strike = 2300
expiration = 1
NAS = 2000  # Number Asset Steps
s0 = 2000

global params
top_barrier = 3  # 3 x strike


def FD_basic(NAS, s0=2000, top_barrier=3, vol=0.2, r=0.5, option_type='e_call', strike=2000, expiration=1):

    #### Inicjalizacja
    dS = top_barrier * strike / NAS                 # dS
    dt = 0.9 / (vol ** 2 * NAS ** 2)                # warunek na dt zapewniający stabilność
    NTS = int(expiration / dt) + 1                  # Number Time Steps
    dt = expiration / NTS                           # dla pewności całkowitości
    V = np.zeros(NAS)                               # wektor, cen opcji (brak pamięci)
    S = np.cumsum(np.ones_like(V) * dS)[::-1]       # Wersja baz pamięci
    ####

    print_it = False
    if print_it is False:
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        print(f'Asset_vec: {S}')

    current_time = expiration - 0 * dt

    if option_type == 'e_call':
        V = np.where(S - strike > 0, S - strike, 0)
    elif option_type == 'e_put':
        V = np.where(strike - S > 0, strike - S, 0)

    v1 = dt / dS ** 2
    v2 = dt / dS
    a_k = 0.5 * vol ** 2 * S ** 2
    b_k = r * S
    c_k = -r
    A_k = v1 * a_k - 0.5 * v2 * b_k
    B_k = -2 * v1 * a_k + dt * c_k
    C_k = v1 * a_k + 0.5 * v2 * b_k

    for t in range(1, NTS):
        current_time -= dt

        V_i = V
        V_down = np.roll(V, -1)
        V_down[-1] = 0
        V_up = np.roll(V, 1)
        V_up[0] = NAS * dS - strike * np.exp(-r * (expiration - current_time))

        #### V^{k+1} = A^k * V^k_{i-1} + (1 + B^k) * V^k_{i} + C^k * V^k_{i+1}
        V = A_k * V_down + (1 + B_k) * V_i + C_k * V_up


    theoretical_price = option_price(s0, strike, expiration, r, vol, 'call')
    numerical_price = interpolate_option_price(S, V, s0)
    diff = np.abs(numerical_price - theoretical_price)

    init = {
        "dS": top_barrier * strike / NAS,               # dS
        "NTS": int(expiration / dt) + 1,                # Number Time Steps
        "dt": expiration / NTS,                         # dla pewności całkowitości
        "V": np.zeros(NAS),                             # wektor, cen opcji (brak pamięci)
        "S": np.cumsum(np.ones_like(V) * dS)[::-1]      # Wersja baz pamięci
    }

    output = {
        'return': numerical_price,
        'theoretical': theoretical_price,
        'diff': diff
    }

    return numerical_price, output, init


def FD_basic_full(NAS, s0=2000, top_barrier=3, vol=0.2, r=0.05, option_type='e_call', strike=2000, expiration=1):
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
    # Obsługa stopy jako liczby (r) lub jako log(1+r)
    if r > 0.2:  # Zakładamy, że stopa >20% jest już w formie log(1+r)
        r_rate = r
    else:
        r_rate = np.log(1 + r)

    #### Inicjalizacja
    dS = top_barrier * strike / NAS                 # dS
    dt = 0.9 / (vol ** 2 * NAS ** 2)                # warunek na dt zapewniający stabilność
    NTS = int(expiration / dt) + 1                  # Number Time Steps
    dt = expiration / NTS                           # dla pewności całkowitości
    V = np.zeros(NAS)                               # wektor cen opcji (brak pamięci)
    S = np.cumsum(np.ones_like(V) * dS)[::-1]       # Wektor cen aktywa (malejąco)

    print_it = False
    if print_it:
        print(f'Krok dS: {dS}')
        print(f'Krok dt: {dt}')
        print(f'NAS: {NAS}')
        print(f'NTS: {NTS}')
        print(f'Asset_vec: {S}')

    # Warunek końcowy (wypłata w terminie wygaśnięcia)
    current_time = expiration - 0 * dt

    # Ustalamy funkcję wypłaty w zależności od typu opcji
    if option_type in ['e_call', 'a_call']:
        V = np.maximum(S - strike, 0)
        payoff = lambda s: np.maximum(s - strike, 0)
    elif option_type in ['e_put', 'a_put']:
        V = np.maximum(strike - S, 0)
        payoff = lambda s: np.maximum(strike - s, 0)

    # Obliczenie współczynników dla schematu różnicowego
    a_k = 0.5 * vol ** 2 * S ** 2
    b_k = r_rate * S
    c_k = -r_rate

    v1 = dt / dS ** 2
    v2 = dt / dS

    A_k = v1 * a_k - 0.5 * v2 * b_k
    B_k = -2 * v1 * a_k + dt * c_k
    C_k = v1 * a_k + 0.5 * v2 * b_k

    # Główna pętla (wstecz w czasie)
    for t in range(1, NTS):
        current_time -= dt

        # Zapisujemy aktualne wartości
        V_i = V.copy()

        # Przygotowanie wektorów dla schematu różnicowego
        V_down = np.roll(V_i, -1)
        V_down[-1] = 0  # Dolny warunek brzegowy

        V_up = np.roll(V_i, 1)

        # Górny warunek brzegowy
        if option_type in ['e_call', 'a_call']:
            V_up[0] = S[0] - strike * np.exp(-r_rate * (expiration - current_time))
        else:  # Put options
            V_up[0] = 0

        # Jawny schemat różnicowy
        V = A_k * V_down + (1 + B_k) * V_i + C_k * V_up

        # Dla opcji amerykańskiej sprawdzamy w każdym kroku, czy wcześniejsze wykonanie jest opłacalne
        if option_type.startswith('a_'):  # Opcje amerykańskie
            intrinsic = payoff(S)
            V = np.maximum(V, intrinsic)

    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call'
    else:
        theoretical_type = 'put'

    # Dla opcji amerykańskich typu call na aktywa bez dywidendy, wartość teoretyczna
    # jest taka sama jak dla opcji europejskiej
    theoretical_price = option_price(s0, strike, expiration, r_rate, vol, theoretical_type)
    numerical_price = interpolate_option_price(S, V, s0)
    diff = np.abs(numerical_price - theoretical_price)

    init = {
        "dS": dS,
        "NTS": NTS,
        "dt": dt,
        "V": np.zeros(NAS),  # Początkowy wektor cen opcji
        "S": S               # Wektor cen aktywa
    }

    output = {
        'return': numerical_price,
        'theoretical': theoretical_price,
        'diff': diff,
        'is_american': option_type.startswith('a_')
    }

    return numerical_price, output, init


def FD_with_memory(NAS, s0=2000, top_barrier=3, vol=0.2, r=0.05, option_type='e_call',
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
    # Obsługa stopy jako liczby (r) lub jako log(1+r)
    if r > 0.2:  # Zakładamy, że stopa >20% jest już w formie log(1+r)
        r_rate = r
    else:
        r_rate = np.log(1 + r)

    #### Inicjalizacja
    dS = top_barrier * strike / NAS                       # dS
    dt = 0.9 / (vol ** 2 * NAS ** 2)                      # warunek na dt zapewniający stabilność
    NTS = int(expiration / dt) + 1                        # Number Time Steps
    dt = expiration / NTS                                 # dla pewności całkowitości

    # Inicjalizacja macierzy cen opcji (wymiary: NAS x NTS) - zapamiętujemy ceny w każdym kroku czasowym
    Grid = np.zeros((NAS, NTS))

    # Wektor cen aktywa (ten sam dla każdego kroku czasowego)
    S = np.cumsum(np.ones(NAS) * dS)[::-1]

    # Wektor czasów
    time_grid = np.linspace(0, expiration, NTS)

    # Debug info
    print_it = False
    if print_it:
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
    b_k = r_rate * S
    c_k = -r_rate

    A_k = v1 * a_k - 0.5 * v2 * b_k
    B_k = -2 * v1 * a_k + dt * c_k
    C_k = v1 * a_k + 0.5 * v2 * b_k

    # Główna pętla (wstecz w czasie)
    for t in range(1, NTS):
        index = (NTS - 1) - t  # Indeks kolumny, którą wypełniamy (idziemy od końca)
        current_time = expiration - t * dt

        # Pobieranie wartości z poprzedniego kroku czasowego
        V_i = Grid[:, index + 1]

        # Przygotowanie wektorów dla schematu różnicowego
        V_down = np.roll(V_i, -1)
        V_down[-1] = 0  # Dolny warunek brzegowy

        V_up = np.roll(V_i, 1)

        # Górny warunek brzegowy
        if option_type in ['e_call', 'a_call']:
            V_up[0] = S[0] - strike * np.exp(-r_rate * (expiration - current_time))
        else:  # Put options
            V_up[0] = 0

        # Jawny schemat różnicowy
        Grid[:, index] = A_k * V_down + (1 + B_k) * V_i + C_k * V_up

        # Dla opcji amerykańskiej sprawdzamy w każdym kroku, czy wcześniejsze wykonanie jest opłacalne
        if option_type.startswith('a_'):  # Opcje amerykańskie
            intrinsic = payoff(S)
            Grid[:, index] = np.maximum(Grid[:, index], intrinsic)

    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call'
    else:
        theoretical_type = 'put'

    theoretical_price = option_price(s0, strike, expiration, r_rate, vol, theoretical_type)
    numerical_price = interpolate_option_price(S, Grid[:, 0], s0)  # Bierzemy pierwszy krok czasowy (czas=0)
    diff = np.abs(numerical_price - theoretical_price)

    # Wartość opcji w każdym kroku czasowym dla ceny s0
    price_evolution = []
    for t in range(NTS):
        interpolated_price = interpolate_option_price(S, Grid[:, t], s0)
        price_evolution.append(interpolated_price)

    # Przygotowanie danych do zwrotu
    grid_data = {
        "Grid": Grid,                   # Pełna macierz cen opcji
        "S": S,                         # Wektor cen aktywa
        "time_grid": time_grid,         # Wektor czasów
        "price_evolution": np.array(price_evolution),  # Ewolucja ceny opcji dla s0
        "dS": dS,
        "dt": dt,
        "NAS": NAS,
        "NTS": NTS
    }

    output = {
        'return': numerical_price,
        'theoretical': theoretical_price,
        'diff': diff,
        'is_american': option_type.startswith('a_')
    }

    return numerical_price, output, grid_data