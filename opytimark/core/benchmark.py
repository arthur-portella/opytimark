"""Benchmark-based class.
"""

from typing import Optional

import numpy as np

import opytimark.utils.exception as e


class Benchmark:
    """A Benchmark class is the root of any benchmarking function.

    It is composed by several properties that defines the traits of a function,
    as well as a non-implemented __call__ method.

    """

    def __init__(
        self,
        name: Optional[str] = "Benchmark",
        dims: Optional[int] = 1,
        continuous: Optional[bool] = False,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = False,
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

        self.name = name
        self.dims = dims
        self.continuous = continuous
        self.convex = convex
        self.differentiable = differentiable
        self.multimodal = multimodal
        self.separable = separable

    @property
    def name(self) -> str:
        """Name of the function."""

        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise e.TypeError("`name` should be a string")

        self._name = name

    @property
    def dims(self) -> int:
        """Number of allowed dimensions."""

        return self._dims

    @dims.setter
    def dims(self, dims: int) -> None:
        if not isinstance(dims, int):
            raise e.TypeError("`dims` should be a integer")
        if dims < -1 or dims == 0:
            raise e.ValueError("`dims` should be >= -1 and different than 0")

        self._dims = dims

    @property
    def continuous(self) -> bool:
        """Whether function is continuous or not."""

        return self._continuous

    @continuous.setter
    def continuous(self, continuous: bool) -> None:
        if not isinstance(continuous, bool):
            raise e.TypeError("`continuous` should be a boolean")

        self._continuous = continuous

    @property
    def convex(self) -> bool:
        """Whether function is convex or not."""

        return self._convex

    @convex.setter
    def convex(self, convex: bool) -> None:
        if not isinstance(convex, bool):
            raise e.TypeError("`convex` should be a boolean")

        self._convex = convex

    @property
    def differentiable(self) -> bool:
        """Whether function is differentiable or not."""

        return self._differentiable

    @differentiable.setter
    def differentiable(self, differentiable: bool) -> None:
        if not isinstance(differentiable, bool):
            raise e.TypeError("`differentiable` should be a boolean")

        self._differentiable = differentiable

    @property
    def multimodal(self) -> bool:
        """Whether function is multimodal or not."""

        return self._multimodal

    @multimodal.setter
    def multimodal(self, multimodal: bool) -> None:
        if not isinstance(multimodal, bool):
            raise e.TypeError("`multimodal` should be a boolean")

        self._multimodal = multimodal

    @property
    def separable(self) -> bool:
        """Whether function is separable or not."""

        return self._separable

    @separable.setter
    def separable(self, separable: bool) -> None:
        if not isinstance(separable, bool):
            raise e.TypeError("`separable` should be a boolean")

        self._separable = separable

    def __call__(self, x: np.array) -> float:
        """This method returns the function's output when the class is called.

        Note that it needs to be implemented in every child class as it is the
        one to hold the benchmarking function logic.

        Args:
            x: An input array for calculating the function's output.

        Returns:
            (float): The benchmarking function output `f(x)`.

        """

        raise NotImplementedError


class MultiObjectiveBenchmark(Benchmark):
    """A MultiObjectiveBenchmark class for multi-objective benchmarks that require an explicit number of objectives.

    """

    def __init__(
        self,
        name: Optional[str] = "MultiObjectiveBenchmark",
        dims: Optional[int] = -1,
        continuous: Optional[bool] = False,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = False,
        multimodal: Optional[bool] = False,
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

        super(MultiObjectiveBenchmark, self).__init__(
            name, dims, continuous, convex, differentiable, multimodal, separable
        )

        self.n_objectives = n_objectives

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""

        return self._n_objectives

    @n_objectives.setter
    def n_objectives(self, n_objectives: int) -> None:
        if not isinstance(n_objectives, int):
            raise e.TypeError("`n_objectives` should be an integer")
        if n_objectives <= 1:
            raise e.ValueError("`n_objectives` should be > 1")

        self._n_objectives = n_objectives
    

class WFGBenchmark(MultiObjectiveBenchmark):
    """A WFGBenchmark class for WFG multi-objective benchmarks

    """

    def __init__(
        self,
        name: Optional[str] = "WFGBenchmark",
        continuous: Optional[bool] = False,
        convex: Optional[bool] = False,
        differentiable: Optional[bool] = False,
        multimodal: Optional[bool] = False,
        separable: Optional[bool] = False,
        n_objectives: Optional[int] = 2,
        k: Optional[int] = 4,
        l: Optional[int] = 20,
    ):
        """Initialization method.

        Args:
            name: Name of the function.
            continuous: Whether the function is continuous.
            convex: Whether the function is convex.
            differentiable: Whether the function is differentiable.
            multimodal: Whether the function is multimodal.
            separable: Whether the function is separable.
            n_objectives: Number of objectives.
            k: Number of position parameters.
            l: Number of distance parameters

        """

        super(WFGBenchmark, self).__init__(
            name=name,
            dims=k + l,
            continuous=continuous,
            convex=convex,
            differentiable=differentiable,
            multimodal=multimodal,
            separable=separable,
            n_objectives=n_objectives,
        )

        self.k = k
        self.l = l

    @property
    def k(self) -> int:
        """Number of position parameters."""

        return self._k

    @k.setter
    def k(self, k: int) -> None:
        if not isinstance(k, int):
            raise e.TypeError("`k` should be an integer")
        if k <= 1:
            raise e.ValueError("`k` should be > 1")

        self._k = k

    @property
    def l(self) -> int:
        """Number of distance parameters."""

        return self._l

    @l.setter
    def l(self, l: int) -> None:
        if not isinstance(l, int):
            raise e.TypeError("`l` should be an integer")
        if l <= 1:
            raise e.ValueError("`l` should be > 1")

        self._l = l

    def normalize(self, z: np.array):
        aux = np.arange(len(z))

        return z / (2 * (aux + 1))
    
    def calculate_x(self, t, A):
        M = len(t)

        x = np.zeros(M)

        for i in range(M - 1):
            x[i] = np.maximum(t[M - 1], A[i]) * (t[i] - 0.5) + 0.5

        x[M - 1] = t[M - 1]

        return x
    
    def calculate_f(self, h, x_M):
        M = len(h)

        f = np.zeros(M)
        
        D = 1
        S = [2 * (i + 1) for i in range(M)]

        for i in range(M):
            f[i] = D * x_M + S[i] * h[i]

        return f
    
    @property
    def lower_bound(self):
        return np.zeros(self.k + self.l)
    
    @property
    def upper_bound(self):
        return 2 * np.arange(1, self.k + self.l + 1)
