"""Multi-objective benchmarking functions.
"""

from typing import Optional

import numpy as np

import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark


class ZDT1(Benchmark):
    """ZDT1 class implements the ZDT1 multi-objective benchmarking function.

    .. math:: f_1(\mathbf{x}) = x_1

    .. math:: g(\mathbf{x}) = 1 + \frac{9}{n-1}\sum_{i=2}^{n} x_i

    .. math:: f_2(\mathbf{x}) = g(\mathbf{x})\left(1-\sqrt{\frac{f_1}{g(\mathbf{x})}}\right)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`f_2 = 1 - \sqrt{f_1}`.

    """

    def __init__(
        self,
        name: Optional[str] = "ZDT1",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
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
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2]`.

        """

        f1 = x[0]

        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)

        f2 = g * (1 - np.sqrt(f1/g))
        
        return np.array([f1, f2])
        

class ZDT2(Benchmark):
    """ZDT2 class implements the ZDT2 multi-objective benchmarking function.

    .. math:: f_1(\mathbf{x}) = x_1

    .. math:: g(\mathbf{x}) = 1 + \frac{9}{n-1}\sum_{i=2}^{n} x_i

    .. math:: f_2(\mathbf{x}) = g(\mathbf{x})\left(1-(\frac{f_1}{g(\mathbf{x})})^2\right)

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`f_2 = 1 - f_1^2`.

    """

    def __init__(
        self,
        name: Optional[str] = "ZDT2",
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

        super(ZDT2, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2]`.

        """

        f1 = x[0]

        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)

        f2 = g * (1 - np.square(f1/g))
        
        return np.array([f1, f2])

class ZDT3(Benchmark):
    """ZDT3 class implements the ZDT3 multi-objective benchmarking function.

    .. math:: f_1(\mathbf{x}) = x_1

    .. math:: g(\mathbf{x}) = 1 + 9 \left( \sum_{i=2}^{n} x_i \right) / (n - 1)

    .. math:: f_2(\mathbf{x}) = g(\mathbf{x}) \left[ 1 - \sqrt{x_1 / g(\mathbf{x})} - \frac{x_1}{g(\mathbf{x})} \sin(10\pi x_1) \right]

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`f_2 = 1 - \sqrt{f_1} - f_1\sin(10\pi f_1)`.

    """

    def __init__(
        self,
        name: Optional[str] = "ZDT3",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = False,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
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

        super(ZDT3, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2]`.

        """

        f1 = x[0]

        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)

        f2 = g * ((1 - np.sqrt(f1 / g))  - (f1 / g) * np.sin(10 * np.pi * f1)) 
        
        return np.array([f1, f2])

class ZDT4(Benchmark):
    """ZDT4 class implements the ZDT4 multi-objective benchmarking function.

    .. math:: f_1(\mathbf{x}) = x_1

    .. math:: g(\mathbf{x}) = 1 + 10(n-1) + \sum_{i=2}^{n} [x_i^2 - 10\cos(4\pi x_i)]

    .. math:: f_2(\mathbf{x}) = g(\mathbf{x})\left\lfloor1-\sqrt{\frac{f_1}{g(\mathbf{x})}}\right\rfloor

    Domain:
        x1 in [0, 1]; xi in [-5, 5] for i = 2, ..., n.

    Pareto-optimal front:
        :math:`f_2 = 1 - \sqrt{f_1}`, when g = 1.0.

    """

    def __init__(
        self,
        name: Optional[str] = "ZDT4",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
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

        super(ZDT4, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2]`.

        """

        f1 = x[0]

        g = 1 + 10 * (len(x) - 1) + np.sum(x[1:]**2 - 10 * np.cos(4 * np.pi * x[1:]))

        f2 = g * (1 - np.sqrt(f1 / g))
        
        return np.array([f1, f2])
        
class ZDT6(Benchmark):
    """ZDT6 class implements the ZDT6 multi-objective benchmarking function.

    .. math:: f_1(\mathbf{x}) = 1 - e^{(-4x_1)} \sin^6(6\pi x_1)

    .. math:: g(\mathbf{x}) = 1 + 9 \left[ \frac{\sum_{i=2}^{n} x_i}{n - 1} \right]^{\frac{1}{4}}

    .. math:: f_2(\mathbf{x}) = g(\mathbf{x}) \left[ 1 - \left( \frac{f_1(x)}{g(x)} \right)^2 \right]

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        An optimal value occurs when f_1 \in [0.28, 1] and f_2 \in [0, 1].

    """

    def __init__(
        self,
        name: Optional[str] = "ZDT6",
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

        super(ZDT6, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2]`.

        """

        f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0])**6

        g = 1 + 9 * (np.sum(x[1:]) / (len(x) - 1))**1/4

        f2 = g * (1 - np.square(f1 / g))
        
        return np.array([f1, f2])
        
