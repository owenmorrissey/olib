from tqdm.auto import tqdm
from joblib import Parallel, parallel_backend, delayed
from .caching import cache_output


# grabbed from https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def parallel_map(
    func,
    objects,
    cache_path=None,
    recompute=False,
    show_progress=True,
):
    "Map func over objects in parallel, and optionally cache the result as a pickle"

    def run():
        with parallel_backend("loky", n_jobs=-1):
            return ProgressParallel(total=len(objects), use_tqdm=show_progress)(
                delayed(func)(obj) for obj in objects
            )

    if cache_path is not None:
        return cache_output(run, (), cache_path, recompute=recompute)
    else:
        return run()
