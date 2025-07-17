import itertools


def flatten(xs_list):
    return list(itertools.chain.from_iterable(xs_list))


def all_empty(xs_list):
    "returns true if xs_list is empty or is a list of empty lists"
    for xs in xs_list:
        if len(xs) > 0:
            return False
    return True
