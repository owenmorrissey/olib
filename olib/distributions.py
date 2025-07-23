import numpy as np
from typing import Dict, List, Optional, Iterable, Any, Union
from itertools import product
from functools import reduce
from collections.abc import MutableMapping

from .lists import flatten_tuples


class DiscreteDistribution(MutableMapping):
    """
    A probability distribution represented as a dictionary where keys are possible
    values and values are their probabilities. Supports proportional and softmax normalization with smoothing.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        data: Optional[Dict[Any, Union[int, float]]] = None,
        normalization_method: str = "proportional",
        smoothing: float = 0.0,
        temperature: float = 1.0,
        domain: Optional[Iterable[Any]] = None,
    ):
        """
        Initialize a discrete distribution.

        Args:
            data: Dictionary of values and their weights/probabilities
            normalization_method: Either "proportional" or "softmax"
            smoothing: Only used for proportional normalization
            temperature: Only used for softmax normalization method
            domain: Optional iterable of all possible keys. Keys not in data will be set to 0.
        """
        self.normalization_method = normalization_method
        self.smoothing = smoothing
        self.temperature = temperature
        self._raw_data = {}
        self._dist = {}

        if data is not None:
            self._raw_data = dict(data)

        self.domain = set(self._raw_data.keys())
        if domain is not None:
            for key in domain:
                self.domain.add(key)

        self._update_dist()

    def get_or_add(self, key):
        self.extend_domain(key)
        return self[key]

    def entropy(self) -> float:
        """Calculate the entropy of the distribution."""
        return -sum(p * np.log2(p) for p in self._dist.values() if p > 0)

    def sample(self, n: int = 1) -> Union[Any, List[Any]]:
        """Sample from the distribution."""
        keys = list(self._dist.keys())
        probs = list(self._dist.values())

        samples = np.random.choice(keys, size=n, p=probs)

        if n == 1:
            return samples[0]
        return samples.tolist()

    def top_k(self, k: int) -> "DiscreteDistribution":
        """Return a new re-normalized distribution with only the top k most probable items."""
        sorted_items = sorted(self._raw_data.items(), key=lambda x: x[1], reverse=True)
        top_items = dict(sorted_items[:k])
        return DiscreteDistribution(
            top_items, self.normalization_method, self.smoothing
        )

    def join_keys(self):
        self._raw_data = {
            "".join(flatten_tuples(key)): val for key, val in self._raw_data.items()
        }
        if self.domain is not None:
            self.domian = set("".join(flatten_tuples(key)) for key in self.domain)
        self._update_dist()
        return self

    def extend_domain(self, key):
        self.domain.add(key)
        self._update_dist()
        return self

    def _update_dist(self):
        """Update the smoothed distribution based on raw data and domain."""
        if self.normalization_method == "softmax":
            data = self._normalize_softmax(self._raw_data, self.temperature)
        elif self.normalization_method == "proportional":
            data = self._raw_data
        else:
            raise ValueError("normalization_method must be 'proportional' or 'softmax'")

        # add zeros for unrepresented but possible events
        zeros = {k: 0 for k in self.domain}
        self._dist = self._normalize_proportional(zeros | data, self.smoothing)

    def _normalize_softmax(
        self, d: Dict[Any, float], temperature: float = 1.0
    ) -> Dict[Any, float]:
        """Apply softmax to a dictionary of values."""
        if temperature <= 0:
            temperature = self.EPSILON  # Avoid division by zero

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
            return {k: (1.0 / len(d)) for k in d.keys()}

        unsmoothed = {k: (v / total) for k, v in d.items()}

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
        return self._dist[key]

    def __setitem__(self, key, value):
        self._raw_data[key] = float(value)
        # If we have a domain and this key isn't in it, add it to the domain
        if self.domain is not None:
            self.domain.add(key)
        self._update_dist()

    def __delitem__(self, key):
        del self._raw_data[key]
        if key in self._dist:
            del self._dist[key]
        self._update_dist()

    def __iter__(self):
        # Return keys in sorted order (by probability, descending)
        sorted_data = self._sort_dict(self._dist)
        return iter(sorted_data)

    def __len__(self):
        return len(self._dist)

    def __repr__(self):
        sorted_data = self._sort_dict(self._dist)
        return f"DiscreteDistribution({dict(sorted_data)})"

    def __str__(self):
        sorted_data = self._sort_dict(self._dist)
        items = [f"{k}: {v:.4f}" for k, v in sorted_data.items()]
        return "{" + ", ".join(items) + "}"

    # Arithmetic operations
    def __add__(self, other):
        """Union of two distributions (adds probabilities for common keys)."""
        if not isinstance(other, DiscreteDistribution):
            raise TypeError("Can only add DiscreteDistribution to DiscreteDistribution")

        result_data = dict(self._raw_data)
        for key, prob in other._raw_data.items():
            if key in result_data:
                result_data[key] += prob
            else:
                result_data[key] = prob

        return DiscreteDistribution(
            result_data, self.normalization_method, self.smoothing
        )

    def __mul__(self, other):
        """Cartesian product of two distributions."""
        if not isinstance(other, DiscreteDistribution):
            raise TypeError(
                f"Cannot multiply DiscreteDistribution by type {type(other)}."
            )

        result_data = {}
        for key1, prob1 in self._raw_data.items():
            for key2, prob2 in other._raw_data.items():
                new_key = (key1, key2)
                likelihood = prob1 * prob2
                if new_key in result_data:
                    result_data[new_key] += likelihood
                else:
                    result_data[new_key] = likelihood

        return DiscreteDistribution(
            result_data, self.normalization_method, self.smoothing
        )

    def update_normalization_method(
        self,
        new_method: Optional[str] = None,
        new_smoothing: Optional[float] = None,
        new_temperature: Optional[float] = None,
    ):
        """Update normalization method and re-normalize. If no arguments are provided, the distribution will be re-normalized with the current parameters."""
        if new_method is not None:
            self.normalization_method = new_method
        if new_smoothing is not None:
            self.smoothing = new_smoothing
        if new_temperature is not None:
            self.temperature = new_temperature
        self._update_dist()

    @staticmethod
    def product(*distributions, concat=False) -> "DiscreteDistribution":
        """
        Compute the Cartesian product of multiple distributions.

        Args:
            *distributions: Variable number of DiscreteDistribution objects
            concat: whether to concatenate resulting keys

        Returns:
            DiscreteDistribution representing the product distribution
        """
        if not distributions:
            return DiscreteDistribution()

        if len(distributions) == 1:
            return DiscreteDistribution(
                distributions[0]._raw_data,
                distributions[0].normalization_method,
                distributions[0].smoothing,
                distributions[0].temperature,
            )

        # Use the normalization method and smoothing from the first distribution
        first_dist = distributions[0]

        result_data = {}

        # Get all combinations using itertools.product
        dist_items = [list(d._raw_data.items()) for d in distributions]

        for combination in product(*dist_items):
            # combination is a tuple of (key, prob) pairs
            keys, probs = zip(*combination)

            new_key = tuple(keys)

            # Multiply probabilities
            likelihood = reduce(lambda x, y: x * y, probs, 1)

            # Add to result
            if new_key in result_data:
                result_data[new_key] += likelihood
            else:
                result_data[new_key] = likelihood

        prod = DiscreteDistribution(
            result_data,
            normalization_method=first_dist.normalization_method,
            smoothing=first_dist.smoothing,
            temperature=first_dist.temperature,
        )

        if concat:
            prod.join_keys()
        return prod


# Example usage and testing
if __name__ == "__main__":
    # Create distributions
    d1 = DiscreteDistribution({"a": 3, "b": 1, "c": 2})
    print("d1:", d1)

    # Create distribution with domain (includes zero-probability events)
    d2 = DiscreteDistribution(
        {"x": 2, "y": 4},
        normalization_method="proportional",
        smoothing=0.1,
        domain=["x", "y", "z", "w"],
    )
    print("d2 with domain [x,y,z,w]:", d2)

    # Test softmax with domain
    d3 = DiscreteDistribution(
        {"a": 1, "b": 3},
        normalization_method="softmax",
        smoothing=0.5,
        domain=["a", "b", "c", "d"],
    )
    print("d3 softmax with domain:", d3)

    # Test addition (union)
    d4 = d1 + DiscreteDistribution({"a": 1, "d": 2})
    print("d1 + d_new:", d4)

    # Test multiplication (product)
    d5 = d1 * DiscreteDistribution({"x": 1, "y": 1})
    print("d1 * d_xy:", d5)

    # Test dictionary-like operations
    print("d1['a']:", d1["a"])
    d1["z"] = 0.5
    print("After adding z:", d1)

    # Test sampling
    print("Sample from d2:", d2.sample(5))

    # test product
    print(DiscreteDistribution.product(d1, d2, d3, d4, d5))
    print(DiscreteDistribution.product(d1, d2, d3, d4, d5, concat=True))
