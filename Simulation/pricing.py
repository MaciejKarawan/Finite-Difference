import numpy as np
from scipy.stats import norm

def option_price(S, K, t, r, sigma, option_type): # Vanilla
    if t <= 0:
        # Dla wygasłej opcji
        if option_type == 'call':
            return np.maximum(S - K, 0)
        else:  # put
            return np.maximum(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_barrier_price(S, K, barrier, t, r, sigma, option_type):
    """
    Wycena opcji barierowej zgodnie z modelem Blacka-Scholesa.

    Parametry:
    S - cena aktualna instrumentu bazowego
    K - cena wykonania
    barrier - poziom bariery
    t - czas do wygaśnięcia w latach
    r - stopa wolna od ryzyka (roczna)
    sigma - zmienność (roczna)
    option_type - typ opcji: 'call_up_and_out', 'put_down_and_out'

    Zwraca:
    Teoretyczną cenę opcji barierowej
    """

    # Stałe z modelu Black-Scholesa
    sqrt_t = np.sqrt(t)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d3 = (np.log(S / barrier) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d4 = (np.log(S / barrier) + (r - 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d5 = (np.log(S / barrier) - (r - 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d6 = (np.log(S / barrier) - (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d7 = (np.log(S*K / barrier**2) - (r - 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d8 = (np.log(S*K / barrier**2) - (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)

    # Zmienne pomocnicze
    alpha = (barrier / S)**(-1 + 2*r / sigma**2)
    beta = (barrier / S)**(1 + 2*r / sigma**2)

    if option_type == 'call_up_and_out':
        if S >= barrier:  # Bariera już przekroczona
            return 0
        value = (S * (norm.cdf(d1) - norm.cdf(d3) - beta * (norm.cdf(d6) - norm.cdf(d8)))
                 - K*np.exp(-r * t) * (norm.cdf(d2) - norm.cdf(d4) - alpha * (norm.cdf(d5) - norm.cdf(d7))))

        return value

    elif option_type == 'put_down_and_out':
        if S <= barrier:
            return 0
        value = (-S * (norm.cdf(d3) - norm.cdf(d1) - beta * (norm.cdf(d8) - norm.cdf(d6)))
                 + K*np.exp(-r * t) * (norm.cdf(d4) - norm.cdf(d2) - alpha * (norm.cdf(d7) - norm.cdf(d5))))

        return value

    else:
        raise ValueError('Zły typ opcji. Dostępne opcje: call_up_and_out, put_down_and_out')


def binary_option_price(S, K, T, r, sigma, option_type='binary_call'):
    if T <= 0:
        # Dla wygasłej opcji
        if option_type == 'binary_call':
            return np.where(S - K > 0, 1, 0)
        elif option_type == 'binary_put':
            return np.where(K - S > 0, 1, 0)
        else:
            raise ValueError("option_type musi być 'call' lub 'put'")

    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'binary_call':
        price = np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'binary_put':
        price = np.exp(-r * T) * (1 - norm.cdf(d2))
    else:
        raise ValueError("option_type musi być 'call' lub 'put'")

    return price


def option_delta(S, K, t, r, sigma, option_type):

    ''' Czy aby na pewno dobrze?
    if t <= 0:
        if option_type == 'call':
            return np.where(S > K, 1.0, 0.0)
        elif option_type == 'put':
             return np.where(S < K, -1.0, 0.0)
    '''

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1.0
    elif option_type == 'binary_call':
        return (np.exp(-r * t) * norm.pdf(d2)) / (sigma * S * np.sqrt(t))
    elif option_type == 'binary_put':
        return - (np.exp(-r * t) * norm.pdf(d2)) / (sigma * S * np.sqrt(t))


def option_gamma(S, K, t, r, sigma, option_type):

    if t <= 0:
        # Dla wygasłej opcji gamma wynosi 0
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type in ['call', 'put']:
        return norm.pdf(d1) / (S * sigma * np.sqrt(t))

    gamma = (np.exp(-r * t) * d1 * norm.pdf(d2)) / (sigma ** 2 * S ** 2 * t)

    if option_type == 'binary_call':
        return -gamma
    elif option_type == 'binary_put':
        return gamma

def d1(S, K, t, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

def d2(S, K, t, r, sigma):
    return d1(S, K, t, r, sigma) - sigma * np.sqrt(t)

'''def gamma_ratio(S, Kv, Kb, t, r, sigma, binary_type):
    d1_v = d1(S, Kv, t, r, sigma)
    d1_b = d1(S, Kb, t, r, sigma)
    d2_b = d1_b - sigma*np.sqrt(t)

    if binary_type == 'binary_call':
        binary_sign = -1
    elif binary_type == 'binary_put':
        binary_sign = 1

    ratio = binary_sign * np.exp(r * t) * sigma * S * np.sqrt(t) * np.exp(0.5 * (d2_b - d1_v) * (d2_b + d1_v)) / d1_b
    return ratio
'''
