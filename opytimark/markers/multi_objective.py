"""Multi-objective benchmarking functions.
"""

from typing import Optional

import numpy as np

import opytimark.utils.exception as e
import opytimark.utils.constants as c
import opytimark.utils.decorator as d
from opytimark.core import Benchmark, MultiObjectiveBenchmark


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

        if (len(x) < 2):
            raise e.ValueError("`n_variables` should be > 1")

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

        if (len(x) < 2):
            raise e.ValueError("`n_variables` should be > 1")

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

        if (len(x) < 2):
            raise e.ValueError("`n_variables` should be > 1")

        f1 = x[0]

        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)

        f2 = g * ((1 - np.sqrt(f1 / g))  - (f1 / g) * np.sin(10 * np.pi * f1)) 
        
        return np.array([f1, f2])

class ZDT4(Benchmark):
    """ZDT4 class implements the ZDT4 multi-objective benchmarking function.

    .. math:: f_1(\mathbf{x}) = x_1

    .. math:: g(\mathbf{x}) = 1 + 10(n-1) + \sum_{i=2}^{n} [x_i^2 - 10\cos(4\pi x_i)]

    .. math:: f_2(\mathbf{x}) = g(\mathbf{x})\left(1-\sqrt{\frac{f_1}{g(\mathbf{x})}}\right)

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
        convex: Optional[bool] = False,
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
        :math:`f_2 = 1 - f_1^2`,
        where :math:`f_1 \in [0.280775, 1]`.
        
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

        if (len(x) < 2):
            raise e.ValueError("`n_variables` should be > 1")

        f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0])**6

        g = 1 + 9 * (np.sum(x[1:]) / (len(x) - 1))**(1/4)

        f2 = g * (1 - np.square(f1 / g))
        
        return np.array([f1, f2])
    

class DTLZ1(MultiObjectiveBenchmark):
    """DTLZ1 class implements the DTLZ1 multi-objective benchmarking function.

    .. math:: f_1(\mathbf{x}) = \frac{1}{2} x_1 x_2 \cdots x_{M-1} (1 + g(\mathbf{x}_M))

    .. math:: f_i(\mathbf{x}) = \frac{1}{2} x_1 \cdots x_{M-i} (1 - x_{M-i+1}) (1 + g(\mathbf{x}_M)),
        \quad i = 2, \ldots, M-1

    .. math:: f_M(\mathbf{x}) = \frac{1}{2} (1 - x_1)(1 + g(\mathbf{x}_M))

    .. math:: g(\mathbf{x}_M) = 100 \left[ k + \sum_{x_i \in \mathbf{x}_M} \left( (x_i - 0.5)^2 - \cos(20 \pi (x_i - 0.5)) \right) \right]

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`\sum_{i=1}^{M} f_i = 0.5`,
        achieved when :math:`g(\mathbf{x}_M) = 0`,
        i.e., :math:`x_i = 0.5` for all :math:`x_i \in \mathbf{x}_M`.
        
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ1",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
        n_objectives: Optional[int] = 3,
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
            n_objectives: Number of objectives.
        """

        super(DTLZ1, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n < M - 1:
            raise e.ValueError("Number of variables must satisfy n >= M - 1")

        f = np.zeros(M)

        k = n - M + 1

        xm = x[n - k:]

        g = 100 * (k + np.sum((xm[:] - 0.5)**2 - np.cos(20 * np.pi * (xm[:] - 0.5))))

        for i in range(M):

            total = 0.5 * (1 + g)
            
            for j in range(M - 1 - i):
                total *= x[j]

            if i > 0:
                total *= (1 - x[M - 1 - i])

            f[i] = total
        
        return f
    

class DTLZ2(MultiObjectiveBenchmark):
    """DTLZ2 class implements the DTLZ2 multi-objective benchmarking function.

    .. math:: f_1 = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}x_j\right)

    .. math:: f_i = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}x_j\right)
        \cdot \sin\!\left(\tfrac{\pi}{2}x_{M-i+1}\right),\quad 1 < i < M

    .. math:: f_M = (1 + g)\sin\!\left(\tfrac{\pi}{2}x_1\right)

    .. math:: g(\mathbf{x}_M) = \sum_{x_i \in \mathbf{x}_M}(x_i - 0.5)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`\sum_{m=1}^{M} f_m^2 = 1.`,
        achieved when :math:`x_i = 0.5` for all :math:`x_i \in \mathbf{x}_M`.
        
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ2",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = True,
        n_objectives: Optional[int] = 3
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
            n_objectives: Number of objectives.
        """

        super(DTLZ2, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n < M - 1:
            raise e.ValueError("Number of variables must satisfy n >= M - 1")

        f = np.zeros(M)

        k = n - M + 1

        xm = x[n - k:]

        g = np.sum((xm[:] - 0.5)**2)

        for i in range(M):

            total = (1 + g)
            
            for j in range(M - 1 - i):
                total *= (np.cos(x[j] * np.pi/2))

            if i > 0:
                total *= (np.sin(x[M - 1 - i] * np.pi/2))

            f[i] = total
        
        return f
    

class DTLZ3(MultiObjectiveBenchmark):

    """DTLZ3 class implements the DTLZ3 multi-objective benchmarking function.

    .. math:: f_1 = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}x_j\right)

    .. math:: f_i = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}x_j\right)
        \cdot \sin\!\left(\tfrac{\pi}{2}x_{M-i+1}\right),\quad 1 < i < M

    .. math:: f_M = (1 + g)\sin\!\left(\tfrac{\pi}{2}x_1\right)

    .. math:: g(\mathbf{x}_M) = 100 \left[ k + \sum_{x_i \in \mathbf{x}_M} \left( (x_i - 0.5)^2 - \cos(20 \pi (x_i - 0.5)) \right) \right]

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`\sum_{m=1}^{M} f_m^2 = 1.`,
        achieved when :math:`x_i = 0.5` for all :math:`x_i \in \mathbf{x}_M`.
        
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ3",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
        n_objectives: Optional[int] = 3
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
            n_objectives: Number of objectives.
        """

        super(DTLZ3, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n < M - 1:
            raise e.ValueError("Number of variables must satisfy n >= M - 1")

        f = np.zeros(M)

        k = n - M + 1

        xm = x[n - k:]

        g = 100 * (k + np.sum((xm[:] - 0.5)**2 - np.cos(20 * np.pi * (xm[:] - 0.5))))

        for i in range(M):

            total = (1 + g)
            
            for j in range(M - 1 - i):
                total *= (np.cos(x[j] * np.pi/2))

            if i > 0:
                total *= (np.sin(x[M - 1 - i] * np.pi/2))

            f[i] = total
        
        return f
    

class DTLZ4(MultiObjectiveBenchmark):

    """DTLZ4 class implements the DTLZ4 multi-objective benchmarking function.

    .. math:: f_1 = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}x_j^\alpha\right)

    .. math:: f_i = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}x_j^\alpha\right)
        \cdot \sin\!\left(\tfrac{\pi}{2}x_{M-i+1}^\alpha\right),\quad 1 < i < M

    .. math:: f_M = (1 + g)\sin\!\left(\tfrac{\pi}{2}x_1^\alpha\right)

    .. math:: g(\mathbf{x}_M) = \sum_{x_i \in \mathbf{x}_M}(x_i - 0.5)^2

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`\sum_{m=1}^{M} f_m^2 = 1.`,
        achieved when :math:`x_i = 0.5` for all :math:`x_i \in \mathbf{x}_M`.
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ4",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
        n_objectives: Optional[int] = 3,
        alpha: Optional[float] = 100,
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
            n_objectives: Number of objectives.
        """

        super(DTLZ4, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )
    
        self.alpha = alpha

    @property
    def alpha(self) -> float:
        """Alpha parameter."""

        return self._alpha

    @alpha.setter
    def alpha(self, alpha) -> None:
        if not isinstance(alpha, (int, float)):
            raise e.TypeError("`alpha` should be a float")
        if alpha <= 0:
            raise e.ValueError("`alpha` should be > 0")

        self._alpha = alpha

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n < M - 1:
            raise e.ValueError("Number of variables must satisfy n >= M - 1")

        f = np.zeros(M)

        k = n - M + 1

        xm = x[n - k:]

        g = np.sum((xm[:] - 0.5)**2)

        for i in range(M):

            total = (1 + g)
            
            for j in range(M - 1 - i):
                total *= (np.cos((x[j]**self.alpha) * np.pi/2))

            if i > 0:
                total *= (np.sin((x[M - 1 - i]**self.alpha) * np.pi/2))

            f[i] = total
        
        return f
    

class DTLZ5(MultiObjectiveBenchmark):
    
    """DTLZ5 class implements the DTLZ5 multi-objective benchmarking function.

    .. math:: f_1 = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}\theta_j\right)

    .. math:: f_i = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\theta_j\right)
        \cdot \sin\!\left(\theta_{M-i+1}\right),\quad 1 < i < M

    .. math:: f_M = (1 + g)\sin\!\left(\theta_1\right)

    .. math:: g(\mathbf{x}_M) = \sum_{x_i \in \mathbf{x}_M}(x_i - 0.5)^2

    .. math:: \theta_1 = \tfrac{\pi}{2}\,x_1,\quad
        \theta_i = \frac{\pi}{4(1+g)}\bigl(1 + 2g\,x_i\bigr),
        \quad i = 2,\ldots,M-1

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`\sum_{m=1}^{M} f_m^2 = 1.`,
        achieved when :math:`x_i = 0.5` for all :math:`x_i \in \mathbf{x}_M`.
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ5",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
        n_objectives: Optional[int] = 3,
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
            n_objectives: Number of objectives.
        """

        super(DTLZ5, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n < M - 1:
            raise e.ValueError("Number of variables must satisfy n >= M - 1")

        f = np.zeros(M)

        k = n - M + 1

        xm = x[n - k:]

        g = np.sum((xm[:] - 0.5)**2)

        theta = np.zeros(M - 1)

        theta[0] = x[0] * np.pi/2

        for i in range(1, M - 1):
            theta[i] = (np.pi / (4 * (1 + g))) * (1 + 2 * g * x[i])

        for i in range(M):

            total = (1 + g)
            
            for j in range(M - 1 - i):
                total *= (np.cos(theta[j]))

            if i > 0:
                total *= (np.sin(theta[M - 1 - i]))

            f[i] = total
        
        return f
    

class DTLZ6(MultiObjectiveBenchmark):
    
    """DTLZ6 class implements the DTLZ6 multi-objective benchmarking function.

    .. math:: f_1 = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\tfrac{\pi}{2}\theta_j\right)

    .. math:: f_i = (1 + g)\prod_{j=1}^{M-i}\cos\!\left(\theta_j\right)
        \cdot \sin\!\left(\theta_{M-i+1}\right),\quad 1 < i < M

    .. math:: f_M = (1 + g)\sin\!\left(\theta_1\right)

    .. math:: g(\mathbf{x}_M) = \sum_{x_i \in \mathbf{x}_M} x_i^{0.1}

    .. math:: \theta_1 = \tfrac{\pi}{2}\,x_1,\quad
        \theta_i = \frac{\pi}{4(1+g)}\bigl(1 + 2g\,x_i\bigr),
        \quad i = 2,\ldots,M-1

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`\sum_{m=1}^{M} f_m^2 = 1.`,
        achieved when :math:`x_i = 0.0` for all :math:`x_i \in \mathbf{x}_M`.
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ6",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = True,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
        n_objectives: Optional[int] = 3,
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
            n_objectives: Number of objectives.
        """

        super(DTLZ6, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n < M - 1:
            raise e.ValueError("Number of variables must satisfy n >= M - 1")

        f = np.zeros(M)

        k = n - M + 1

        xm = x[n - k:]

        g = np.sum(xm[:]**0.1)

        theta = np.zeros(M - 1)

        theta[0] = x[0] * np.pi/2

        for i in range(1, M - 1):
            theta[i] = (np.pi / (4 * (1 + g))) * (1 + 2 * g * x[i])

        for i in range(M):

            total = (1 + g)
            
            for j in range(M - 1 - i):
                total *= (np.cos(theta[j]))

            if i > 0:
                total *= (np.sin(theta[M - 1 - i]))

            f[i] = total
        
        return f


class DTLZ7(MultiObjectiveBenchmark):
    
    """DTLZ7 class implements the DTLZ7 multi-objective benchmarking function.

    .. math:: f_i(x_i) = x_i,\quad i = 1,\ldots,M-1
    
    .. math:: f_M = (1 + g)\,h(f_1,\ldots,f_{M-1},\,g)

    .. math:: g(\mathbf{x}_M) = 1 + \frac{9}{k}\sum_{x_i \in \mathbf{x}_M} x_i

    .. math:: h = M - \sum_{i=1}^{M-1}\left[\frac{f_i}{1+g}
        \bigl(1 + \sin(3\pi f_i)\bigr)\right]

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        :math:`2^(M−1)` disconnected regions,
        achieved when :math:`x_i = 0.0` for all :math:`x_i \in \mathbf{x}_M`.
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ7",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = False,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = False,
        n_objectives: Optional[int] = 3,
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
            n_objectives: Number of objectives.
        """

        super(DTLZ7, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n < M:
            raise e.ValueError("Number of variables must satisfy n >= M")

        f = np.zeros(M)

        k = n - M + 1

        xm = x[n - k:]

        g = 1 + (9 / k) * np.sum(xm[:])

        for i in range(M - 1):
            f[i] = x[i]

        h = M - np.sum(f[:M - 1]/(1 + g) * (1 + np.sin(3 * np.pi * f[:M - 1])))

        f[M - 1] = (1 + g) * h
        
        return f


class DTLZ8(MultiObjectiveBenchmark):
    
    """DTLZ8 class implements the DTLZ8 constrained multi-objective benchmarking function.

    .. math:: f_j(\mathbf{x}) = \frac{1}{\lfloor n/M \rfloor} \sum_{i=\lfloor (j-1)n/M \rfloor}^{\lfloor jn/M \rfloor} x_i, \quad j = 1,2,\ldots,M
    
    Subject to:

    .. math:: g_j(\mathbf{x}) = f_M(\mathbf{x}) + 4f_j(\mathbf{x}) - 1 \geq 0, \quad j = 1,2,\ldots,M-1

    .. math:: g_M(\mathbf{x}) = 2f_M(\mathbf{x}) + \min_{\substack{i,j=1 \\ i \neq j}}^{M-1}\left[f_i(\mathbf{x}) + f_j(\mathbf{x})\right] - 1 \geq 0

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Constraints:
        The problem has :math:`M` inequality constraints that define a disconnected feasible Pareto-optimal front.    

    Pareto-optimal front:
        Obtained when all constraints are satisfied. The resulting front is discontinuous and presents several disconnected regions.
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ8",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = True,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
        n_objectives: Optional[int] = 3,
        penalty: Optional[float] = 100.0,
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
            n_objectives: Number of objectives.
            penalty: Penalty coefficient applied to solutions that
            violate the constraints.
        """

        super(DTLZ8, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

        self.penalty = penalty

    @property
    def penalty(self) -> float:
        """Penalty parameter."""

        return self._penalty

    @penalty.setter
    def penalty(self, penalty) -> None:
        if not isinstance(penalty, (int, float)):
            raise e.TypeError("`penalty` should be a float or integer")
        if penalty < 0:
            raise e.ValueError("`penalty` should be >= 0")

        self._penalty = penalty    

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n <= M:
            raise e.ValueError("Number of variables must satisfy n > M")
        
        if M < 3:
            raise e.ValueError("Number of objectives must be at least 3")

        f = np.zeros(M)

        constraints = np.zeros(M)

        for i in range(M):
            f[i] = (1 / np.floor(n / M).astype(int)) * np.sum(x[np.floor(i * n / M).astype(int) : np.floor((i + 1) * n / M).astype(int)])

        for i in range(M - 1):
            constraints[i] = f[-1] + 4 * f[i] - 1

        min_sum = np.inf

        for j in range(M - 1):
            for k in range(j + 1, M - 1):
                min_sum = min(min_sum, f[j] + f[k])

        constraints[-1] = 2 * f[-1] + min_sum - 1        

        violation = 0

        for i in range(M):
            violation += max(0, -constraints[i]) ** 2
        
        f += self.penalty * violation

        return f
    

class DTLZ9(MultiObjectiveBenchmark):
    
    """DTLZ9 class implements the DTLZ9 constrained multi-objective benchmarking function.

    .. math:: f_j(\mathbf{x}) = \sum_{i=\lfloor (j-1)n/M \rfloor}^{\lfloor jn/M \rfloor} x_i^{0.1}, \quad j = 1,2,\ldots,M
    
    Subject to:

    .. math:: g_j(\mathbf{x}) = f_M^2(\mathbf{x}) + f_j^2(\mathbf{x}) - 1 \geq 0, \quad j = 1,2,\ldots,M-1

    Domain:
        The function is commonly evaluated using :math:`x_i \in [0, 1]`.

    Pareto-optimal front:
        Obtained when all constraints are satisfied. The resulting front is a curve with :math:`f_1 = f_2 = \\cdots = f_{M-1}`.
    """

    def __init__(
        self,
        name: Optional[str] = "DTLZ9",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = True,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = False,
        multimodal: Optional[bool] = True,
        separable: Optional[bool] = True,
        n_objectives: Optional[int] = 3,
        penalty: Optional[float] = 100.0,
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
            n_objectives: Number of objectives.
            penalty: Penalty coefficient applied to solutions that
            violate the constraints.
        """

        super(DTLZ9, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable, n_objectives
        )

        self.penalty = penalty

    @property
    def penalty(self) -> float:
        """Penalty parameter."""

        return self._penalty

    @penalty.setter
    def penalty(self, penalty) -> None:
        if not isinstance(penalty, (int, float)):
            raise e.TypeError("`penalty` should be a float or integer")
        if penalty < 0:
            raise e.ValueError("`penalty` should be >= 0")

        self._penalty = penalty    

    @d.check_exact_dimension
    def __call__(self, x: np.array) -> np.ndarray:
        """This method returns the function's output when the class is called.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (np.array): The benchmarking function outputs `[f1, f2, ..., fm]`.

        """

        M = self.n_objectives
        n = len(x)

        if n <= M:
            raise e.ValueError("Number of variables must satisfy n > M")
        
        if M < 3:
            raise e.ValueError("Number of objectives must be at least 3")

        f = np.zeros(M)

        constraints = np.zeros(M - 1)

        for i in range(M):
            f[i] = np.sum(x[np.floor(i * n / M).astype(int) : np.floor((i + 1) * n / M).astype(int)] ** 0.1)

        for i in range(M - 1):
            constraints[i] = (f[-1] ** 2) + (f[i] ** 2) - 1

        violation = 0

        for i in range(M - 1):
            violation += max(0, -constraints[i]) ** 2
        
        f += self.penalty * violation

        return f