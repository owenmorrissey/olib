import itertools
import operator
import functools


def flatten(xs_list, method="reduce"):
    # fastest
    if method == "reduce":
        return list(functools.reduce(operator.iconcat, xs_list, []))
    elif method == "chain":
        return list(itertools.chain.from_iterable(xs_list))
    else:
        raise NotImplementedError(f"method: {method} is not implemented.")


def all_empty(xs_list):
    "returns true if xs_list is empty or is a list of empty lists"
    for xs in xs_list:
        if len(xs) > 0:
            return False
    return True


def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x
