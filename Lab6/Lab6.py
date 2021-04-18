import sympy as sp
from sympy.plotting import plot


def main():
    xp = sp.Symbol('x')
    k = 10
    m = 1.8
    p = [0.0, 0.41, 0.79, 1.13, 1.46, 1.76, 2.04, 2.3, 2.55, 2.79, 3.01]
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y = [p[i] + ((-1)**k)*m for i in range(len(p))]

    print("x = ", x)
    print("\ny = ", y)
    print("\nИнтерполяционный многочлен методом Лагранжа:")
    print(lagrange_method(x, xp, y))
    result = lagrange_method(x, xp, y).subs('x', 0.47)
    print("\nЗначение функции в точке ", xp)
    print(result)
    print()
    print("Интерполяционный многочлен методом Ньютона:")
    print(newton_method(x, y, xp))
    result = newton_method(x, y, xp).subs('x', 0.47)
    xp = 0.47
    print("\nЗначение функции в точке ", xp)
    print(result)


def func_omega(x, xp=0, exclude=None, top_value=None):
    result = 1
    if top_value is None:
        top_value = len(x)
    for i in range(top_value):
        if i == exclude:
            continue
        result *= (xp - x[i])
    return result


def lagrange_method(x, xp, y):
    result = 0
    for i in range(len(x)):
        result += y[i]*(func_omega(x=x, xp=xp, exclude=i)/func_omega(x=x, xp=x[i], exclude=i))
    result = sp.expand(result)
    return result
        

def func_f(x, y, top_idx=None):
    result = 0
    for k in range(top_idx + 1):
        result += y[k]/func_omega(x=x, xp=x[k], exclude=k, top_value=top_idx + 1)
    return result


def newton_method(x, y, xp):
    result = 0
    for i in range(len(x)):
        result += func_f(x, y, i)*func_omega(x=x, xp=xp, top_value=i)
    result = sp.expand(result)
    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program was stopped with error")
