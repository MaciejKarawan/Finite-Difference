from FD import *
from FD_KO import *
from IF1.projekt2.src.FD_KO import FDB_with_memory
from pricing import *
from visualization import *
import numpy as np

### Args
vol = 0.20
vol2 = [0.2, 0.3]
r = np.log(1 + 0.05)
option_type = 'e_put'
strike = 2300
expiration = 1
NAS = 500  # Number Asset Steps
s0 = 2194.05
barrier = 2800
top_barrier = 3

#_, output_basic_wo_memory, init_basic_wo_memory = FD_wo_memory(NAS, s0, top_barrier, vol, r, option_type, strike, expiration)
_, output_basic_w_memory, init_basic_w_memory, _ = FD_with_memory(NAS, s0, top_barrier, vol, r, option_type, strike, expiration)

print(output_basic_w_memory['diff'])
print(output_basic_w_memory['return'])
print(output_basic_w_memory['theoretical'])



#_, data_knock, _ = FDB_wo_memory(NAS, s0, vol2, r, option_type, strike, expiration, barrier)
#print(data_knock['return'])


'''def prepare_fd_comparison_data(NAS_list, s0=2000, top_barrier=3, vol=0.2, r=0.05,
                               option_type='e_call', strike=2000, expiration=1):
    """
    Przygotowuje dane do porównania metod FD (bez pamięci i z pamięcią)

    Parametry:
    - NAS_list: lista wartości NAS (rozmiarów siatki) do przetestowania
    - pozostałe parametry jak w FD_wo_memory i FD_with_memory

    Zwraca:
    - dict z danymi dla metody bez pamięci
    - dict z danymi dla metody z pamięcią
    - listę grid_sizes
    """
    results_wo_memory = {
        'return': [],
        'theoretic': [],
        'diff': [],
        'time': []
    }

    results_with_memory = {
        'return': [],
        'theoretic': [],
        'diff': [],
        'time': []
    }

    for NAS in NAS_list:
        # Obliczenia dla metody bez pamięci
        _, output_wo, init_wo = FD_wo_memory(
            NAS=NAS, s0=s0, top_barrier=top_barrier, vol=vol, r=r,
            option_type=option_type, strike=strike, expiration=expiration
        )

        # Obliczenia dla metody z pamięcią
        _, output_with, init_with, _ = FD_with_memory(
            NAS=NAS, s0=s0, top_barrier=top_barrier, vol=vol, r=r,
            option_type=option_type, strike=strike, expiration=expiration
        )

        # Zapis wyników
        results_wo_memory['return'].append(output_wo['return'])
        results_wo_memory['theoretic'].append(output_wo['theoretical'])
        results_wo_memory['diff'].append(output_wo['diff'])
        results_wo_memory['time'].append(init_wo['time'])

        results_with_memory['return'].append(output_with['return'])
        results_with_memory['theoretic'].append(output_with['theoretical'])
        results_with_memory['diff'].append(output_with['diff'])
        results_with_memory['time'].append(init_with['time'])

    return results_wo_memory, results_with_memory, NAS_list

grids = [10, 100, 200, 400, 500, 1000, 2000]

# Przygotowanie danych
results_wo, results_with, grid_sizes = prepare_fd_comparison_data(
    NAS_list=grids,
    s0=s0,
    strike=strike,
    option_type=option_type
)

# Wizualizacja
plot_fd_comparison(
    results_wo_memory=results_wo,
    results_with_memory=results_with,
    grid_sizes=grid_sizes,
    title=f'Porównanie metod FD dla opcji {option_type}\nS0={s0}, K={strike}',
    show_time=True
)'''