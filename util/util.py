def is_real_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_positive_real_number(s: str):
    try:
        return float(s) > 0
    except ValueError:
        return False


def is_natural_number(s: str):
    if s.isdigit():
        return int(s) >= 1
    return False


def is_positive_natural_number(s: str):
    return is_natural_number(s) and int(s) > 0
