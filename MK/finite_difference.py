import numpy as np
import sys
import os
from interpolation import interpolate_option_price

sciezka = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'projekt1', 'src'))
sys.path.append(sciezka)
from pricing import *

np.set_printoptions(
    precision=2,
    linewidth=200,
    suppress=True
)

############################ Explicit Finite Difference Method

### Args
vol = 0.20
r = 0.05
option_type = 'call'
strike = 2300
expiration = 1
NAS = 2000                          # Number Asset Steps
top_barrier = 3                     # 3 x strike


#### Setup

dS = top_barrier * strike / NAS
dt = 0.9 / (vol**2 * NAS**2)                            # warunek na dt zapewniający stabilność
NTS = int(expiration / dt) + 1                          # Number Time Steps
dt = expiration / NTS                                   # dla pewności całkowitości
#Grid = np.zeros((NAS, NTS))                            # macierz cen opcji (jeżeli chcemy pamiętać ceny)
Grid = np.zeros(NAS)                                    # wektor, cen opcji (brak pamięci)
#S = np.cumsum(np.ones_like(Grid[:, -1]) * dS)[::-1]    # Wektor cen aktywa (wersja z pamięcią)
S = np.cumsum(np.ones_like(Grid) * dS)[::-1]     # Wersja baz pamięci
####

print(f'Krok dS: {dS}')
print(f'Krok dt: {dt}')
print(f'NAS: {NAS}')
print(f'NTS: {NTS}')
print(f'Asset_vec: {S}')

current_time = expiration - 0 * dt
#Grid[:, -1] = np.where(S - strike > 0, S - strike, 0)          # Wersja z pamięcią
Grid = np.where(S - strike > 0, S - strike, 0)                  # Bez

v1 = dt / dS**2
v2 = dt / dS

for t in range(1, NTS):
    index = (NTS - 1) - t
    current_time -= dt

    ''' Wersja z pamięcią 
    ### V_{i-1}, V_{i}, V_{i+1}
    V_i = Grid[:, index + 1]
    V_down = np.roll(Grid[:, index + 1], -1)
    V_down[-1] = 0
    V_up = np.roll(Grid[:, index + 1], 1)
    V_up[0] = NAS * dS - strike * np.exp(-r * (expiration - current_time))
    a_k = 0.5 * vol**2 * S**2
    b_k = r * S
    c_k = r
    '''

    V_i = Grid
    V_down = np.roll(Grid, -1)
    V_down[-1] = 0
    V_up = np.roll(Grid, 1)
    V_up[0] = NAS * dS - strike * np.exp(-r * (expiration - current_time))
    a_k = 0.5 * vol ** 2 * S ** 2
    b_k = r * S
    c_k = r


    A_k = v1 * a_k - 0.5 * v2 * b_k
    B_k = -2 * v1 * a_k + dt * c_k
    C_k = v1 * a_k + 0.5 * v2 * b_k

    #### V^{k+1} = A^k * V^k_{i-1} + (1 + B^k) * V^k_{i} + C^k * V^k_{i+1}
    Grid = A_k * V_down + (1 + B_k) * V_i + C_k * V_up


theoretical_price = option_price(2000, strike, expiration, r, vol, 'call')
numerical_price = interpolate_option_price(S, Grid, 2000)



print(f'Teoretyczna wartość: {theoretical_price} \n'
      f'Finite-Difference: {numerical_price} \n'
      f'Różnica: {np.abs(numerical_price - theoretical_price)}')





