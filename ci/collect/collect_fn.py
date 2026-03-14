import builtins


def max(values):
    return builtins.max(values)


def min(values):
    return builtins.min(values)


def mean(values):
    return builtins.sum(values) / len(values)


def sum(values):
    return builtins.sum(values)


def alltrue(values):
    return builtins.all(values)


def allfalse(values):
    return not builtins.any(values)


def anytrue(values):
    return builtins.any(values)


def anyfalse(values):
    return not builtins.all(values)
