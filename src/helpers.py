""" Various misc helper methods. """


def clamp(num: float, minimum: float, maximum: float) -> float:
    """ Returns the closest value to num within the range [minimum, maximum]. """
    if num < minimum:
        return minimum
    if num > maximum:
        return maximum
    return num


def round_to(value: float, incr: float) -> float:
    """ Rounds value to the nearest inc. """
    return incr * round(value / incr)
