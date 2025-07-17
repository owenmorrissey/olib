import pandas as pd
from pathlib import Path


def cache_output(func, args, pkl_path, kwargs={}, recompute=False):
    "Pickle the output of func(*args, *kwargs)"
    cache_path = Path(pkl_path)
    if cache_path.exists() and not recompute:
        result = pd.read_pickle(cache_path)
    else:
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        result = func(*args, **kwargs)
        pd.to_pickle(result, cache_path)
    return result


def memo(f):
    "Memoize function f, whose args must all be hashable."
    cache = {}

    def fmemo(*args):
        if args not in cache:
            cache[args] = f(*args)
        return cache[args]

    fmemo.cache = cache
    return fmemo
