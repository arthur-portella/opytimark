"""Transformations."""

import numpy as np

import opytimark.utils.exception as e

# The primary parameters y and y1,...,yabs(y) aways have domain[0, 1].
#    A, B, C, alpha and beta are constants. For b_param, y' is a vector of
#    secundary parameters (of domain [0, 1]), and u is a reduction function.


def b_poly(y, alpha):
    if alpha <= 0 or alpha == 1:
        raise e.ValueError("`alpha` should be > 0 and != 1")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    return np.clip(y**alpha, 0.0, 1.0)


def b_flat(y, A, B, C):
    if not (0 <= A <= 1 and 0 <= B <= 1 and 0 <= C <= 1):
        raise e.ValueError("`A`,`B` and `C` should be >= 0 and <= 1")

    if B >= C:
        raise e.ValueError("`B` should be < `C`")

    if B == 0 and (A != 0 or C == 1):
        raise e.ValueError("since `B` = 0, `A` should be = 0 and `C` should be != 1")

    if C == 1 and (A != 1 or B == 0):
        raise e.ValueError("since `C` = 1, `A` should be = 1 and `B` should be != 0")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    # return A + np.minimum(0, np.floor(y - B)) * ((A * (B - y)) / B) - np.minimum(0, np.floor(C - y)) * (((1 - A) * (y - C)) / (1 - C))

    # outra alternativa, divisão por 0 é igual a 0:
    new_B = 0.0 if B == 0 else np.minimum(0, np.floor(y - B)) * ((A * (B - y)) / B)
    new_C = (
        0.0
        if C == 1
        else np.minimum(0, np.floor(C - y)) * ((1 - A) * (y - C) / (1 - C))
    )
    return np.clip(A + new_B - new_C, 0.0, 1.0)


def b_param(y, _y, A, B, C):
    if not (0 < A < 1):
        raise e.ValueError("`A` should be > 0 and < 1")

    if not (0 < B and B < C):
        raise e.ValueError("`B` should be > 0 and < C")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    u = np.mean(_y)
    v = A - (1 - 2 * u) * np.abs(np.floor(0.5 - u) + A)
    return np.clip(y ** (B + (C - B) * v), 0.0, 1.0)


def s_linear(y, A):
    if not (0 < A < 1):
        raise e.ValueError("`A` should be > 0 and < 1")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    return np.clip((np.abs(y - A) / np.abs(np.floor(A - y) + A)), 0.0, 1.0)


def s_decept(y, A, B, C):
    if not (0 < A < 1 and 0 < B < 1 and 0 < C < 1):
        raise e.ValueError("`A`,`B` and `C` should be > 0 and < 1")

    if A - B <= 0:
        raise e.ValueError("`A - B` should be > 0")

    if A + B >= 1:
        raise e.ValueError("`A + B` should be < 1")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    return np.clip(
        1
        + (np.abs(y - A) - B)
        * (
            (np.floor(y - A + B) * (1 - C + ((A - B) / B))) / (A - B)
            + (np.floor(A + B - y) * (1 - C + ((1 - A - B) / B))) / (1 - A - B)
            + (1 / B)
        ),
        0.0,
        1.0,
    )


def s_multi(y, A, B, C):
    if not (A > 0):
        raise e.ValueError("`A` should be > 0")

    if not (B >= 0):
        raise e.ValueError("`B` should be >= 0")

    if not ((4 * A + 2) * np.pi >= 4 * B):
        raise e.ValueError("`(4*A + 2)*np.pi` should be >= `4*B`")

    if not (0 < C < 1):
        raise e.ValueError("`C` should be > 0 and < 1")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    return np.clip(
        (
            1
            + np.cos(
                (4 * A + 2)
                * np.pi
                * (0.5 - ((np.abs(y - C)) / (2 * (np.floor(C - y) + C))))
            )
            + 4 * B * ((np.abs(y - C)) / (2 * (np.floor(C - y) + C))) ** 2
        )
        / (B + 2),
        0.0,
        1.0,
    )


def r_sum(y, w):
    if len(w) != len(y):
        raise e.SizeError("`w` should be same size as `y`")

    if not (all(w_i > 0 for w_i in w[: len(y)])):
        raise e.ValueError("`w_i` should be > 0")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    return np.clip(np.sum(w * y) / np.sum(w), 0.0, 1.0)


def r_nonsep(y, A):
    if not (1 <= A <= len(y)):
        raise e.ValueError("`A` should be >= 1 and <= |y|")

    if not (len(y) % A == 0):
        raise e.ValueError("`A` should be a divisor of |y|")

    if np.any(y < 0) or np.any(y > 1):
        raise e.ValueError("`y` should be in [0,1]")

    n = len(y)
    dif = sum(np.sum(np.abs(y - np.roll(y, -k))) for k in range(1, A))
    numerator = np.sum(y) + dif

    ceil_A = np.ceil(A / 2)
    denominator = (n / A) * ceil_A * (1 + 2 * A - 2 * ceil_A)

    return np.clip(numerator / denominator, 0.0, 1.0)
