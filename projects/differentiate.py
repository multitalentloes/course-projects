import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    u_out = []
    u_out.append((u[1]-u[0])/dt)
    for idx in range(1, len(u) - 1):
        u_out.append((u[idx+1]-u[idx-1])/(2*dt))
    u_out.append((u[len(u)-1]-u[len(u)-2])/dt)
    return u_out
        

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    u_out = np.zeros((len(u)))
    u_out[1:len(u)-1] = (u[2:len(u)] - u[0:len(u)-2]) / (2*dt)
    u_out[0] = (u[1]-u[0])/dt
    u_out[len(u)-1] = (u[len(u)-1]-u[len(u)-2])/dt
    return u_out

def backward_diff(u: np.ndarray, dt: float) -> np.ndarray:
    u_out = []
    for idx in range(1, len(u)):
        u_out.append((u[idx]-u[idx-1])/(dt))
    return u_out

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    exact = [2*t_i for t_i in t]
    assert np.allclose(du1, du2)
    assert np.allclose(du1[1:len(t)-1], exact[1:len(t)-1])

    # first and last element deviate from the exact solution
    assert not abs(exact[0] - du1[0]) < 0.0001
    assert not abs(exact[len(du1)-1] - du1[len(du1)-1]) < 0.0001 

if __name__ == '__main__':
    test_differentiate()
    