"""
WIP: the idea is that in jupyter notebooks and colabs,
I ususally just copy-paste imports for the same 10 libraries.
This is supposed to do that in a one-liner.
"""


def import_standard():
    pkgs = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborntqdm",
    ]

    import math
    import sys
    import os
    import itertools

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from pathlib import Path
    from tqdm import tqdm


def colab_setup():
    pass
