""" Various misc helper methods. """


def clamp(num: float, minimum: float, maximum: float) -> float:
    """ Returns the closest value to num within the range [minimum, maximum]. """
    if num < minimum:
        return minimum
    if num > maximum:
        return maximum
    return num
