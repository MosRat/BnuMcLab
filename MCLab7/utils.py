import contextlib
from typing import Callable, Iterable, Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from matplotlib.colors import Normalize
from concurrent.futures import ProcessPoolExecutor, as_completed
from warnings import filterwarnings
filterwarnings('ignore')


@contextlib.contextmanager
def multiprocess_as_completed(process: int,
                              func: Callable,
                              params,
                              names=None,
                              *args, **kwargs) -> Tuple[Iterator, Dict]:
    """
    多进程运行
    """
    with ProcessPoolExecutor(max_workers=process) as pool:
        if names is None:
            names = params
        if isinstance(params[0], dict):
            futures = {pool.submit(func, *args, **param, **kwargs): name for name, param in zip(names, params)}
        elif isinstance(params[0], (list, tuple)):
            futures = {pool.submit(func, *args, *param, **kwargs): name for name, param in zip(names, params)}
        else:
            futures = {pool.submit(func, *args, param, **kwargs): name for name, param in zip(names, params)}
        yield as_completed(futures), futures
