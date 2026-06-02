"""Multi-objective benchmarking functions.
"""

from typing import Optional

import numpy as np

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class ZDT1(Benchmark):
    """Forrester class implements the Forrester's benchmarking function.

    .. math:: f(x) = (6x - 2)^2 sin(12x - 4)

    Domain:
        The function is commonly evaluated using :math:`x \in [0, 1]`.

    Global Minima:
        :math:`f(x^*) \\approx -5.9932767166446155 \mid x^* \\approx (0.75)`.

    """

    def __init__(
        self,
        name: Optional[str] = "ZDT1",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            dims: Number of allowed dimensions.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.

        """

        super(ZDT1, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.array:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        f1 = x[0]

        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)

        f2 = g * (1 - np.sqrt(f1/g)) if f1 != 0 else g
        
        return np.array([f1, f2])
        
