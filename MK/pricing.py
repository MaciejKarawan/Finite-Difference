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
    Wycena opcji barierowej zgodnie z modelem Wilmotta.

    Parametry:
    S - cena aktualna instrumentu bazowego
    K - cena wykonania
    barrier - poziom bariery
    t - czas do wygaśnięcia w latach
    r - stopa wolna od ryzyka (roczna)
    sigma - zmienność (roczna)
    option_type - typ opcji: 'call_up_and_out', 'call_down_and_out',
                  'call_up_and_in', 'call_down_and_in', 'put_up_and_out',
                  'put_down_and_out', 'put_up_and_in', 'put_down_and_in'

    Zwraca:
    Teoretyczną cenę opcji barierowej
    """
    if t <= 0:
        return 0

    # Stałe z modelu Black-Scholesa
    sqrt_t = np.sqrt(t)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    # Zmienne pomocnicze dla modelu barierowego
    mu = (r - 0.5 * sigma ** 2) / (sigma ** 2)
    lambda_param = np.sqrt(mu ** 2 + 2 * r / (sigma ** 2))

    # Potencjał do odbicia
    x1 = np.log(S / barrier) / (sigma * sqrt_t) + (mu + 1) * sigma * sqrt_t
    y1 = np.log(barrier / S) / (sigma * sqrt_t) + (mu + 1) * sigma * sqrt_t
    y2 = np.log(barrier / S) / (sigma * sqrt_t) + mu * sigma * sqrt_t
    z = np.log(barrier / S) / (sigma * sqrt_t) + lambda_param * sigma * sqrt_t

    # Funkcje pomocnicze dla różnych typów opcji barierowych
    def vanilla_call():
        return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    def vanilla_put():
        return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    if option_type == 'call_up_and_out':
        if S >= barrier:  # Bariera już przekroczona
            return 0

        # Cena opcji call up-and-out
        term1 = vanilla_call()
        term2 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(y1)
        term3 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(y1 - sigma * sqrt_t)

        return term1 - term2 + term3

    elif option_type == 'call_down_and_out':
        if S <= barrier:  # Bariera już przekroczona
            return 0

        # Cena opcji call down-and-out
        if barrier <= K:
            return vanilla_call() - (S * (barrier / S) ** (2 * mu + 2) * norm.cdf(-y1) -
                                     K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(-y1 + sigma * sqrt_t))
        else:
            # Dla bariery powyżej ceny wykonania
            term1 = S * norm.cdf(x1)
            term2 = K * np.exp(-r * t) * norm.cdf(x1 - sigma * sqrt_t)
            term3 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(y2)
            term4 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(y2 - sigma * sqrt_t)

            return term1 - term2 - term3 + term4

    elif option_type == 'call_up_and_in':
        if S >= barrier:  # Bariera już przekroczona
            return vanilla_call()

        # Cena opcji call up-and-in
        term1 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(y1)
        term2 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(y1 - sigma * sqrt_t)

        return term1 - term2

    elif option_type == 'call_down_and_in':
        if S <= barrier:  # Bariera już przekroczona
            return vanilla_call()

        # Cena opcji call down-and-in
        if barrier <= K:
            return S * (barrier / S) ** (2 * mu + 2) * norm.cdf(-y1) - K * np.exp(-r * t) * (barrier / S) ** (
                        2 * mu) * norm.cdf(-y1 + sigma * sqrt_t)
        else:
            term1 = vanilla_call()
            term2 = S * norm.cdf(x1)
            term3 = K * np.exp(-r * t) * norm.cdf(x1 - sigma * sqrt_t)
            term4 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(y2)
            term5 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(y2 - sigma * sqrt_t)

            return term1 - (term2 - term3 - term4 + term5)

    elif option_type == 'put_up_and_out':
        if S >= barrier:  # Bariera już przekroczona
            return 0

        # Cena opcji put up-and-out
        if barrier >= K:
            return vanilla_put() - (K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(y1 - sigma * sqrt_t) -
                                    S * (barrier / S) ** (2 * mu + 2) * norm.cdf(y1))
        else:
            # Dla bariery poniżej ceny wykonania
            term1 = K * np.exp(-r * t) * norm.cdf(-x1 + sigma * sqrt_t)
            term2 = S * norm.cdf(-x1)
            term3 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(-y2 + sigma * sqrt_t)
            term4 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(-y2)

            return term1 - term2 - term3 + term4

    elif option_type == 'put_down_and_out':
        if S <= barrier:  # Bariera już przekroczona
            return 0

        # Cena opcji put down-and-out
        term1 = vanilla_put()
        term2 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(-y1 + sigma * sqrt_t)
        term3 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(-y1)

        return term1 - term2 + term3

    elif option_type == 'put_up_and_in':
        if S >= barrier:  # Bariera już przekroczona
            return vanilla_put()

        # Cena opcji put up-and-in
        if barrier >= K:
            return K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(y1 - sigma * sqrt_t) - S * (
                        barrier / S) ** (2 * mu + 2) * norm.cdf(y1)
        else:
            term1 = vanilla_put()
            term2 = K * np.exp(-r * t) * norm.cdf(-x1 + sigma * sqrt_t)
            term3 = S * norm.cdf(-x1)
            term4 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(-y2 + sigma * sqrt_t)
            term5 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(-y2)

            return term1 - (term2 - term3 - term4 + term5)

    elif option_type == 'put_down_and_in':
        if S <= barrier:  # Bariera już przekroczona
            return vanilla_put()

        # Cena opcji put down-and-in
        term1 = K * np.exp(-r * t) * (barrier / S) ** (2 * mu) * norm.cdf(-y1 + sigma * sqrt_t)
        term2 = S * (barrier / S) ** (2 * mu + 2) * norm.cdf(-y1)

        return term1 - term2

    else:
        raise ValueError('Zły typ opcji')


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
