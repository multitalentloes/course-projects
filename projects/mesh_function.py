from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt

def mesh_function(f: Callable[[float], float], t: np.ndarray) -> np.ndarray:
    return np.array([f(t_i) for t_i in t])

def func(t: float) -> float:
    if t >= 0.0 and t <= 3.0:
        return np.exp(-t)
    elif t > 3.0 and t <= 4:
        return np.exp(-3*t)
    else:
        raise RuntimeError(f"provided a t={t} outside of the valid domain!")

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)

def plot_mesh_function(dt: float):
    n = int(round((4/dt)+1))
    xs = np.linspace(0, 4, n)
    ys = mesh_function(func, xs)

    plt.plot(xs, ys, marker="o", markersize="2", label="f")
    plt.xlabel("t (within range [0, 4])")
    plt.ylabel("f(t)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_mesh_function()

    plot_mesh_function(0.1)
    plot_mesh_function(0.01)
