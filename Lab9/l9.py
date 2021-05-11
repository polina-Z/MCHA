from matplotlib import pylab
from scipy.integrate import odeint as od
import numpy as np

EPS = 0.001

a = 0
b = 1
y0 = 0


def to_fixed(numObj, digits=0):
    if isinstance(numObj, list):
        return [to_fixed(i, digits) for i in numObj]
    return f"{numObj:.{digits}f}"


def f(y, x):
    return (1.0 * (1 - y ** 2)) / ((1 + 1.3) * (x ** 2) + (y ** 2) + 1)


def runge_kutta(x, step):
    if x <= a:
        return y0
    prev_x = x - step
    prev_y = runge_kutta(prev_x, step)
    k1 = f(prev_y, prev_x)
    k2 = f(prev_y + (step / 2) * k1, prev_x + step / 2)
    k3 = f(prev_y + (step / 2) * k2, prev_x + step / 2)
    k4 = f(prev_y + step * k3, x)
    return prev_y + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def find_step():
    h0 = EPS ** (1 / 4)
    n = int((b - a) // h0)
    if n % 2 != 0:
        n += 1
    while check_step(n):
        n = n // 4 * 2
    while not check_step(n):
        n += 2
    return (b - a) / n


def check_step(n):
    h = (b - a) / n
    y2 = runge_kutta(a + 2 * h, h)
    y2e = runge_kutta(a + 2 * h, h * 2)
    eps = (1 / 15) * abs(y2 - y2e)
    return eps < EPS


def exact(x):
    sol = od(f, y0, [a, x])
    return sol[1][0]


def next_modified_y(x, y_prev, h):
    return y_prev + h*f(x+h/2, y_prev+h/2*f(x, y_prev))


def euler(h):
    x = a
    y = 0
    y_k = []
    while x <= b:
        y_k.append(y)
        y = next_modified_y(x, y, h)
        x += h
    return y_k


def main():
    print("Исходные данные:")
    print(f"y = {1.3} * (1 - y ** 2)) / ((1 + {1.0}) * (x ** 2) + (y ** 2) + 1")
    print(f"y(0) = {y0}")
    print(f"\nИнтервал: [{a}, {b}]")
    print(f"Погрешность: {EPS}")
    print()
    step = find_step()
    print("Шаг итерирования: ", step)
    xlist = np.arange(a, b + step, step)
    runge_kutta_points = []
    euler_points = []
    euler_points = euler(step)
    exact_points = []
    x = a
    while x <= b:
        r = runge_kutta(x, step)
        r2 = exact(x)
        runge_kutta_points.append(r)
        exact_points.append(r2)
        x += step
    print(f"Значения функции в точках методом Элера: {to_fixed(euler_points, 4)}")
    print(f"Значения функции в точках методом Рунге-Кутта: {to_fixed(runge_kutta_points, 4)}")
    print(f"Точные значения функции: {to_fixed(exact_points, 4)}")    
    pylab.cla()
    pylab.plot (xlist, exact_points, label = "точное решение", color = (0, 1, 0))
    pylab.plot (xlist, euler_points, label = "кривая методом Эйлера", color = (1, 0, 0))
    pylab.plot (xlist, runge_kutta_points, label = "кривая методом Рунге-Кутта ", color = (0, 0, 1))
    pylab.grid(True)
    pylab.legend()
    pylab.savefig("lab9.png")
    pylab.show()
    

if __name__ == '__main__':
    main()
