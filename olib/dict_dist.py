import numpy as np
from typing import Dict, List, Callable, Iterable, Any, Union
from itertools import product
from functools import reduce
from collections.abc import MutableMapping


class DictDistribution(MutableMapping):
    """
    A probability distribution represented as a dictionary where keys are possible
    values and values are their probabilities.
    """

    def __init__(
        self,
        data: Dict[Any, Union[int, float]] = None,
        normalization_method: str = "proportional",
        smoothing: float = 0.0,
        domain: Iterable[Any] = None,
    ):
        """
        Initialize a dictionary distribution.

        Args:
            data: Dictionary of values and their weights/probabilities
            normalization_method: Either "proportional" or "softmax"
            smoothing: Smoothing parameter (acts as (temperature-1) for softmax)
            domain: Optional iterable of all possible keys. Keys not in data will be set to 0.
        """
        self.normalization_method = normalization_method
        self.smoothing = smoothing
        self.temperature = smoothing
        self.domain = set(domain) if domain is not None else None
        self._data = {}

        if data is not None:
            self._data = dict(data)

        # Add domain keys with value 0 if they're not already present
        if self.domain is not None:
            for key in self.domain:
                if key not in self._data:
                    self._data[key] = 0.0

        self._normalize()

    def _normalize(self):
        """Normalize the distribution based on the chosen method."""
        if not self._data:
            return

        if self.normalization_method == "softmax":
            self._data = self._dict_softmax(self._data, self.smoothing)
        elif self.normalization_method == "proportional":
            self._data = self._normalize_proportional(self._data, self.smoothing)
        else:
            raise ValueError("normalization_method must be 'proportional' or 'softmax'")

    def _dict_softmax(
        self, d: Dict[Any, float], temperature: float = 1.0
    ) -> Dict[Any, float]:
        """Apply softmax to a dictionary of values."""
        if temperature <= 0:
            temperature = 1e-8  # Avoid division by zero

        # Apply temperature scaling
        scaled_values = {k: v / temperature for k, v in d.items()}
        # Subtract max for numerical stability
        max_v = max(scaled_values.values())
        # Calculate exp of each value
        exp_values = {k: np.exp(v - max_v) for k, v in scaled_values.items()}
        # Calculate sum for normalization
        sum_exp = sum(exp_values.values())
        # Normalize to ensure values sum to 1
        return {k: v / sum_exp for k, v in exp_values.items()}

    def _normalize_proportional(
        self, d: Dict[Any, float], smoothing: float = 0.0
    ) -> Dict[Any, float]:
        """Normalize proportionally with optional smoothing."""
        total = sum(d.values())
        if total == 0:
            # Uniform distribution if all values are 0
            return {k: 1.0 / len(d) for k in d.keys()}

        unsmoothed = {k: v / total for k, v in d.items()}

        if smoothing == 0:
            return unsmoothed

        # Apply Laplace smoothing
        smoothed = {
            k: (v + smoothing) / (1 + smoothing * len(d)) for k, v in unsmoothed.items()
        }
        return smoothed

    def _sort_dict(self, d: Dict[Any, float]) -> Dict[Any, float]:
        """Sort dictionary by values in descending order."""
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

    # MutableMapping interface
    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = float(value)
        # If we have a domain and this key isn't in it, add it to the domain
        if self.domain is not None:
            self.domain.add(key)
        self._normalize()

    def __delitem__(self, key):
        del self._data[key]
        self._normalize()

    def __iter__(self):
        # Return keys in sorted order (by probability, descending)
        sorted_data = self._sort_dict(self._data)
        return iter(sorted_data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        sorted_data = self._sort_dict(self._data)
        return f"DictDistribution({dict(sorted_data)})"

    def __str__(self):
        sorted_data = self._sort_dict(self._data)
        items = [f"{k}: {v:.4f}" for k, v in sorted_data.items()]
        return "{" + ", ".join(items) + "}"

    # Arithmetic operations
    def __add__(self, other):
        """Union of two distributions (adds probabilities for common keys)."""
        if not isinstance(other, DictDistribution):
            raise TypeError("Can only add DictDistribution to DictDistribution")

        result_data = dict(self._data)
        for key, prob in other._data.items():
            if key in result_data:
                result_data[key] += prob
            else:
                result_data[key] = prob

        return DictDistribution(result_data, self.normalization_method, self.smoothing)

    def __mul__(self, other):
        """Cartesian product of two distributions."""
        if not isinstance(other, DictDistribution):
            raise TypeError("Can only multiply DictDistribution by DictDistribution")

        result_data = {}
        for key1, prob1 in self._data.items():
            for key2, prob2 in other._data.items():
                new_key = str(key1) + str(key2)  # Concatenate keys
                likelihood = prob1 * prob2
                if new_key in result_data:
                    result_data[new_key] += likelihood
                else:
                    result_data[new_key] = likelihood

        return DictDistribution(result_data, self.normalization_method, self.smoothing)

    # Additional utility methods
    def update_smoothing(self, new_smoothing: float):
        """Update smoothing parameter and re-normalize."""
        self.smoothing = new_smoothing
        self._normalize()

    def update_normalization_method(self, new_method: str):
        """Update normalization method and re-normalize."""
        self.normalization_method = new_method
        self._normalize()

    def entropy(self) -> float:
        """Calculate the entropy of the distribution."""
        return -sum(p * np.log2(p) for p in self._data.values() if p > 0)

    def sample(self, n: int = 1) -> Union[Any, List[Any]]:
        """Sample from the distribution."""
        keys = list(self._data.keys())
        probs = list(self._data.values())

        samples = np.random.choice(keys, size=n, p=probs)

        if n == 1:
            return samples[0]
        return samples.tolist()

    def top_k(self, k: int) -> "DictDistribution":
        """Return a new distribution with only the top k most probable items."""
        sorted_items = sorted(self._data.items(), key=lambda x: x[1], reverse=True)
        top_items = dict(sorted_items[:k])
        return DictDistribution(top_items, self.normalization_method, self.smoothing)

    @staticmethod
    def product(*distributions) -> "DictDistribution":
        """
        Compute the Cartesian product of multiple distributions.

        Args:
            *distributions: Variable number of DictDistribution objects

        Returns:
            DictDistribution representing the product distribution
        """
        if not distributions:
            return DictDistribution()

        if len(distributions) == 1:
            return DictDistribution(
                distributions[0]._data,
                distributions[0].normalization_method,
                distributions[0].smoothing,
            )

        # Use the normalization method and smoothing from the first distribution
        first_dist = distributions[0]

        result_data = {}

        # Get all combinations using itertools.product
        dist_items = [list(d._data.items()) for d in distributions]

        for combination in product(*dist_items):
            # combination is a tuple of (key, prob) pairs
            keys, probs = zip(*combination)

            # Concatenate keys to form new key
            new_key = "".join(str(k) for k in keys)

            # Multiply probabilities
            likelihood = reduce(lambda x, y: x * y, probs, 1)

            # Add to result
            if new_key in result_data:
                result_data[new_key] += likelihood
            else:
                result_data[new_key] = likelihood

        return DictDistribution(
            result_data, first_dist.normalization_method, first_dist.smoothing
        )

    # Example usage and testing


if __name__ == "__main__":
    # Create distributions
    d1 = DictDistribution({"a": 3, "b": 1, "c": 2})
    print("d1:", d1)

    # Create distribution with domain (includes zero-probability events)
    d2 = DictDistribution(
        {"x": 2, "y": 4},
        normalization_method="proportional",
        smoothing=0.1,
        domain=["x", "y", "z", "w"],
    )
    print("d2 with domain [x,y,z,w]:", d2)

    # Test softmax with domain
    d3 = DictDistribution(
        {"a": 1, "b": 3},
        normalization_method="softmax",
        smoothing=0.5,
        domain=["a", "b", "c", "d"],
    )
    print("d3 softmax with domain:", d3)

    # Test addition (union)
    d4 = d1 + DictDistribution({"a": 1, "d": 2})
    print("d1 + d_new:", d4)

    # Test multiplication (product)
    d5 = d1 * DictDistribution({"x": 1, "y": 1})
    print("d1 * d_xy:", d5)

    # Test dictionary-like operations
    print("d1['a']:", d1["a"])
    d1["z"] = 0.5
    print("After adding z:", d1)

    # Test sampling
    print("Sample from d2:", d2.sample(5))
