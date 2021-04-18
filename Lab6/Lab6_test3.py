import sympy as sp
from sympy.plotting import plot


def main():
    xp = sp.Symbol('x')
    k = 10
    m = 1.8
    x = [1, 3, 4]
    y = [6, 24, 45]

    print("x = ", x)
    print("\ny = ", y)
    print("\nИнтерполяционный многочлен методом Лагранжа:")
    print(lagrange_method(x, xp, y))
    print("Интерполяционный многочлен методом Ньютона:")
    print(newton_method(x, y, xp))



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
