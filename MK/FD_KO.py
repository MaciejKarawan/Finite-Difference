from interpolation import interpolate_option_price
from pricing import *
import time

np.set_printoptions(
    precision=2,
    linewidth=200,
    suppress=True
)


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
    - barrier_type: typ bariery ('up' dla górnej, 'down' dla dolnej)

    Zwraca:
    - numerical_price: wyceniona wartość opcji
    - output: słownik z dodatkowymi informacjami
    - init: słownik z parametrami inicjalizacyjnymi
    """

    barrier_type = 'up' if option_type in ['e_call', 'a_call'] else 'down'

    #### Inicjalizacja
    # Określenie zakresu siatki w zależności od typu bariery i opcji
    if barrier_type == 'up':
        max_S = barrier
        min_S = 0
        dS = max_S / NAS
    elif barrier_type == 'down':
        min_S = barrier
        max_S = 4 * strike
        dS = (max_S - min_S) / NAS

    # Warunek na dt zapewniający stabilność dla maksymalnej zmienności
    dt = 0.9 / (vol[1] ** 2 * NAS ** 2)
    NTS = int(expiration / dt) + 1  # Number Time Steps
    dt = expiration / NTS  # dla pewności całkowitości
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

    for t in range(1, NTS):
        current_time = expiration - t * dt
        V_i = V.copy()

        ##### Obliczanie gammy i zmienności (sprawdzić poprawność interpolacji)
        V_up = np.roll(V_i, 1)
        V_down = np.roll(V_i, -1)
        V_up[0] = 2 * V_i[0] - V_i[1]
        V_down[-1] = 2 * V_i[-2] - V_i[-1]
        gamma = (V_up - 2 * V_i + V_down) / (dS ** 2)

        # Ustalanie zmienności
        sigma = np.where(gamma > 0, vol[0], vol[1])


        ### Obliczenie współczynników dla schematu różnicowego
        a_k = 0.5 * sigma ** 2 * S ** 2
        b_k = r * S
        c_k = -r * np.ones_like(S)

        v1 = dt / dS ** 2
        v2 = dt / dS

        A = v1 * a_k + 0.5 * v2 * b_k
        B = -2 * v1 * a_k + dt * c_k
        C = v1 * a_k - 0.5 * v2 * b_k

        ### Jawny schemat różnicowy z warunkami brzegowymi
        # Wewnętrzne punkty
        V[1:-1] = A[1:-1] * V_i[0:-2] + (1 + B[1:-1]) * V_i[1:-1] + C[1:-1] * V_i[2:]

        # Warunki brzegowe, takie same bez względu na typ opcji/bariery
        V[0] = 0
        V[-1] = 0

        ### Dla opcji amerykańskiej sprawdzamy wcześniejsze wykonanie
        if option_type.startswith('a_'):
            intrinsic_value = payoff(S)
            V = np.where(barrier_mask, np.maximum(V, intrinsic_value), 0)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call'
    else:
        theoretical_type = 'put'

    if type(s0) is int or type(s0) is float:
        # Sprawdzamy, czy cena początkowa nie jest za barierą
        if (barrier_type == 'up' and s0 >= barrier) or (barrier_type == 'down' and s0 <= barrier):
            numerical_price = 0
        else:
            theoretical_price = option_barrier_price(s0, strike, barrier, expiration, r, vol[0], 'call_up_and_out') # HERE: dodać GAMME w t=0
            numerical_price = interpolate_option_price(S, V, s0)

        output = {
            'return': numerical_price,
            'theoretical': theoretical_price,
            'diff': np.abs(numerical_price - theoretical_price),
        }
    else:
        numerical_price = s0
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
        "vol_range": vol
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

    # Określenie zakresu siatki w zależności od typu bariery i opcji
    if barrier_type == 'up':
        max_S = barrier
        min_S = 0
        dS = max_S / NAS
    else:  # Bariera dolna (down-and-out)
        min_S = barrier
        max_S = 4 * strike
        dS = (max_S - min_S) / NAS

    # Inicjalizacja
    dt = 0.9 / (vol[1] ** 2 * NAS ** 2)
    NTS = int(expiration / dt) + 1                          # Number Time Steps
    dt = expiration / NTS                                   # dla pewności całkowitości
    Grid = np.zeros((NAS + 1, NTS))                         # Inicjalizacja macierzy cen opcji (wymiary: (NAS+1) x NTS)
    Gamma_Grid = np.zeros((NAS + 1, NTS))                   # Dodane: macierz gamma do śledzenia wartości gamma w każdym kroku czasowym
    Early_Exercise = np.zeros((NAS + 1, NTS))               # Dodane: macierz do śledzenia wcześniejszych wykonań dla opcji amerykańskich
    S = np.linspace(max_S, min_S, NAS + 1)                  # Tworzymy siatkę cen aktywa - od najwyższej do najniższej
    time_grid = np.linspace(0, expiration, NTS)        # Wektor czasów

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

    # Główna pętla (wstecz w czasie)
    for t in range(1, NTS):
        index = (NTS - 1) - t                   # Indeks kolumny, którą wypełniamy (idziemy od końca)
        current_time = expiration - t * dt
        V_i = Grid[:, index + 1]                # Pobieranie wartości z poprzedniego kroku czasowego

        # Obliczamy gammę dla każdego węzła (druga pochodna) - wektorowo
        V_up = np.roll(V_i, 1)
        V_down = np.roll(V_i, -1)

        # Korekta dla krańcowych wartości
        V_up[0] = V_i[0]  # Pierwszy element - brak sąsiada z góry
        V_down[-1] = V_i[-1]  # Ostatni element - brak sąsiada z dołu

        # Obliczenie gammy
        gamma = np.zeros_like(S)
        gamma[1:-1] = (V_up[1:-1] - 2 * V_i[1:-1] + V_down[1:-1]) / (dS ** 2)

        # Zapisujemy wartości gammy do macierzy
        Gamma_Grid[:, index] = gamma

        # Wybieramy zmienność na podstawie znaku gammy - wektorowo
        sigma[:] = np.where(gamma > 0, vol[0], vol[1])

        # Obliczenie współczynników dla schematu różnicowego
        a_k = 0.5 * sigma ** 2 * S ** 2
        b_k = r * S
        c_k = -r * np.ones_like(S)

        v1 = dt / dS ** 2
        v2 = dt / dS

        A = v1 * a_k + 0.5 * v2 * b_k
        B = -2 * v1 * a_k + dt * c_k
        C = v1 * a_k - 0.5 * v2 * b_k

        # Jawny schemat różnicowy z warunkami brzegowymi
        # Wewnętrzne punkty
        Grid[1:-1, index] = A[1:-1] * V_i[0:-2] + (1 + B[1:-1]) * V_i[1:-1] + C[1:-1] * V_i[2:]

        # Warunki brzegowe
        if option_type in ['e_call', 'a_call']:
            if barrier_type == 'up':
                Grid[0, index] = 0  # Na górnej barierze wartość opcji wynosi 0
                Grid[-1, index] = 0  # Dolny warunek brzegowy dla S=0
            else:  # barrier_type == 'down'
                Grid[0, index] = 0  # Na dolnej barierze wartość opcji wynosi 0
                Grid[-1, index] = S[-1] - strike * np.exp(-r * (expiration - current_time))  # Asymptota dla małych S
        elif option_type in ['e_put', 'a_put']:
            if barrier_type == 'up':
                Grid[0, index] = 0  # Na górnej barierze wartość opcji wynosi 0
                Grid[-1, index] = strike * np.exp(-r * (expiration - current_time))  # Asymptota dla S=0
            else:  # barrier_type == 'down'
                Grid[0, index] = 0  # Na dolnej barierze wartość opcji wynosi 0
                Grid[-1, index] = 0  # Górny warunek brzegowy dla małych S

        # Dla opcji amerykańskiej sprawdzamy wcześniejsze wykonanie - wektorowo
        if option_type.startswith('a_'):  # Opcje amerykańskie
            intrinsic_value = payoff(S)
            early_exercise_mask = np.zeros_like(S, dtype=bool)

            # Sprawdzamy, gdzie opłaca się wcześniejsze wykonanie
            early_exercise_mask = (intrinsic_value > Grid[:, index]) & barrier_mask

            # Aktualizujemy wartości opcji
            Grid[early_exercise_mask, index] = intrinsic_value[early_exercise_mask]

            # Oznaczamy miejsca wcześniejszego wykonania
            Early_Exercise[early_exercise_mask, index] = 1

    # Obliczamy gammę dla pierwszego kroku czasowego (t=0)
    V_i = Grid[:, 0]
    V_up = np.roll(V_i, 1)
    V_down = np.roll(V_i, -1)
    V_up[0] = V_i[0]
    V_down[-1] = V_i[-1]
    Gamma_Grid[1:-1, 0] = (V_up[1:-1] - 2 * V_i[1:-1] + V_down[1:-1]) / (dS ** 2)

    # Obliczanie wartości teoretycznej i numerycznej
    if option_type in ['e_call', 'a_call']:
        theoretical_type = 'call'
    else:
        theoretical_type = 'put'

    if type(s0) is int or type(s0) is float:
        # Sprawdzamy, czy cena początkowa nie jest za barierą
        if (barrier_type == 'up' and s0 >= barrier) or (barrier_type == 'down' and s0 <= barrier):
            numerical_price = 0  # Za barierą wartość opcji wynosi 0
            price_evolution = np.zeros(NTS)
        else:
            # Dla opcji barierowych nie ma prostego wzoru na cenę teoretyczną
            theoretical_price = None  # Można by dodać wzór analityczny jeśli dostępny
            numerical_price = interpolate_option_price(S, Grid[:, 0], s0)  # Bierzemy pierwszy krok czasowy (czas=0)

            # Wartość opcji w każdym kroku czasowym dla ceny s0
            price_evolution = []
            for t in range(NTS):
                interpolated_price = interpolate_option_price(S, Grid[:, t], s0)
                price_evolution.append(interpolated_price)

        output = {
            'return': numerical_price,
            'theoretical': theoretical_price if theoretical_price is not None else 'N/A',
            'diff': np.abs(numerical_price - theoretical_price) if theoretical_price is not None else 'N/A',
            "price_evolution": np.array(price_evolution),  # Ewolucja ceny opcji dla s0
        }

    else:
        numerical_price = Grid[:, 0]
        output = {
            'return': numerical_price
        }

    # Przygotowanie danych do zwrotu
    init = {
        "S": S,  # Wektor cen aktywa
        "time_grid": time_grid,  # Wektor czasów
        "dS": dS,
        "dt": dt,
        "NAS": NAS,
        "NTS": NTS,
        "barrier": barrier,
        "barrier_type": barrier_type,
        "vol_range": vol
    }
    Grid_data = {
        'Grid': Grid,
        'Gamma': Gamma_Grid,  # Pełna macierz gamma
        'Sigma': sigma,  # Wektor zmienności
        'Early_Exercise': Early_Exercise  # Macierz wcześniejszych wykonań
    }

    return numerical_price, output, init, Grid_data