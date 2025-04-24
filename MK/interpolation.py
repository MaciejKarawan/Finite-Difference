def interpolate_option_price(S_vector, option_prices, target_price):
    # Znajdź indeksy, między którymi znajduje się target_price
    # Ponieważ S_vector jest malejący (z góry na dół)
    for i in range(len(S_vector) - 1):
        if S_vector[i] >= target_price and S_vector[i + 1] <= target_price:
            # Interpolacja liniowa
            x0, x1 = S_vector[i], S_vector[i + 1]
            y0, y1 = option_prices[i], option_prices[i + 1]

            # Wzór na interpolację liniową: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            return y0 + (y1 - y0) * (target_price - x0) / (x1 - x0)

    # Jeśli target_price jest poza zakresem, zwróć najbliższą wartość
    if target_price > S_vector[0]:
        return option_prices[0]
    else:
        return option_prices[-1]