import numpy as np
from scipy.stats import norm


def option_price(S, K, t, r, sigma, option_type):
    """Oblicza cenę opcji europejskiej metodą Blacka-Scholesa."""
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


def binary_option_price(S, K, T, r, sigma, option_type='call'):
    """
    Wycenia europejską opcję binarną.

    Parameters:
    -----------
    S : float
        Cena aktywa bazowego.
    K : float
        Cena wykonania.
    T : float
        Czas do wygaśnięcia (w latach).
    r : float
        Stopa wolna od ryzyka (roczna).
    sigma : float
        Zmienność (roczna).
    option_type : str, optional
        Typ opcji: 'call' lub 'put'. Domyślnie 'call'.

    Returns:
    --------
    price : float
        Cena opcji binarnej.
    """
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type.lower() == 'call':
        price = np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type musi być 'call' lub 'put'")

    return price


def option_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Oblicza greki dla opcji europejskiej.

    Parameters:
    -----------
    S : float
        Cena aktywa bazowego.
    K : float
        Cena wykonania.
    T : float
        Czas do wygaśnięcia (w latach).
    r : float
        Stopa wolna od ryzyka (roczna).
    sigma : float
        Zmienność (roczna).
    option_type : str, optional
        Typ opcji: 'call' lub 'put'. Domyślnie 'call'.

    Returns:
    --------
    greeks : dict
        Słownik z wartościami greck: delta, gamma, theta, vega, rho.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (taka sama dla call i put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta
    if option_type.lower() == 'call':
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    # Vega (taka sama dla call i put)
    vega = S * np.sqrt(T) * norm.pdf(d1)

    # Rho
    if option_type.lower() == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

def option_delta(S, K, t, r, sigma, option_type):
    """Oblicza deltę opcji europejskiej."""
    if t <= 0:
        # Dla wygasłej opcji
        if option_type == 'call':
            return np.where(S > K, 1.0, 0.0)
        else:  # put
             return np.where(S < K, -1.0, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    if option_type == 'call':
        return norm.cdf(d1)
    else:  # put
        return norm.cdf(d1) - 1.0