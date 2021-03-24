import numpy as np


Iters = 0


def input():
    expr = np.poly1d([1.0, 38.4621, 364.594, 914.196])
    return (expr)


def SturmSeq(f):
    arr = []
    arr.append(f)
    arr.append(np.polyder(f))

    while True:
        fn = -np.polydiv(arr[-2], arr[-1])[1]
        if (fn.order > 0 or abs(fn[0]) > 0.0):
            arr.append(fn)
        else:
            break
    return arr


def N(stseq, x, eps, f):
    if (abs(f(x)) < eps):
        raise ValueError("Number in N() is a root")
    ans = 0
    for i in range(1, len(stseq)):
        if (stseq[i](x) == 0.0):
            raise ValueError("SturmSeq[i] is zero")
        if (stseq[i - 1](x) * stseq[i](x) < 0):
            ans += 1
    return ans


def GetBounds(f, a, b, eps, Sturm):
    if ((abs(f(a)) < eps) or (abs(f(b)) < eps)):
        raise ValueError("Bounds contain root")
    if (N(Sturm, a, eps, f) - N(Sturm, b, eps, f) == 0):
        return []
    if (N(Sturm, a, eps, f) - N(Sturm, b, eps, f) > 1):
        while True:
            M = a + (b - a) / (1.5 + np.random.random())
            if (abs(f(M)) > eps):
                break
        return GetBounds(f, a, M, eps, Sturm) + GetBounds(f, M, b, eps, Sturm)
    if (b - a < eps):
        print("Warning: Bounds are too small")
    return [(a, b)]


def BinarySearch(L, R, f, eps):
    global Iters
    iters += 1
    M = (L + R) / 2
    if (R - L < eps):
        return M
    if (f(L) * f(M) <= 0):
        return BinarySearch(L, M)
    elif (f(R) * f(M) <= 0):
        return BinarySearch(M, R)
    else:
        raise RuntimeError("Something went wrong in BinarySearch")


def SecantFirst(L, R, f, eps):
    global Iters
    fder2 = np.polyder(f, 2)
    if (f(R) * fder2(R) > 0):
        (oldx, x) = (R, L)
    elif (f(L) * fder2(L) > 0):
        (oldx, x) = (L, R)
    else:
        raise ValueError("Bad bounds in first Secant method")
        (oldx, x) = (R, L)
    t = oldx
    while (abs(x - oldx) > EPS):
        # while (abs(f(x)) > EPS):
        iters += 1
        oldx = x
        x = x - f(x) * (t - x) / (f(t) - f(x))
    return x


def SecantSecond(L, R, f, eps):
    global Iters
    (x, oldx) = (L, R)
    while (abs(x - oldx) > eps):
        # while (abs(f(x)) > EPS):
        iters += 1
        oldx = x
        x = L - f(L) * (R - L) / (f(R) - f(L))
        if (f(L) * f(x) <= 0):
            R = x
        elif (f(R) * f(x) <= 0):
            L = x
        else:
            raise RuntimeError("Something is wrong in second Secant method")
    return x


def Newton(L, R, f, eps):
    global Iters
    fder = np.polyder(f)
    fder2 = np.polyder(f, 2)
    if (f(L) * fder2(L) > 0):
        (oldx, x) = (R, L)
    elif (f(R) * fder2(R) > 0):
        (oldx, x) = (L, R)
    else:
        raise ValueError("Bad bounds in Newton method")
        (oldx, x) = (L, R)
    while (abs(x - oldx) > EPS):
        # while (abs(f(x)) > EPS):
        iters += 1
        oldx = x
        x = x - f(x) / fder(x)
    return x


def test(method, bounds, f, eps):
    global Iters
    for i in range(len(bounds)):
        iters = 0
        try:
            str = method(*bounds[i], f, eps)
            if (not str is None):
                str = "{:.4f}".format(str)
            print("{} via {} method (with {} iterations)".format(str, method.__name__, iters))
        except Exception as ex:
            print("ERROR: {} - in {} method (with {} iterations)".format(ex, method.__name__, iters))


def main():
    np.random.seed(42)
    print("Nonlinear equations \n")
    eps = 10.0 ** -5

    (f) = input()
    print(f)

    sturm = SturmSeq(f)

    print("Amount of roots on [-10, 10]:")
    print(N(sturm, -10, eps, f) - N(sturm, 10, eps, f))

    bounds = GetBounds(f, -10, 10, eps, sturm)
    print("Roots are in bounds:")
    print(bounds)

    np.set_printoptions(suppress=True, precision=4, floatmode="fixed")

    print()
    test(BinarySearch, bounds, f, eps)
    print()
    test(SecantFirst, bounds, f, eps)
    print()
    test(SecantSecond, bounds, f, eps)
    print()
    test(Newton, bounds, f, eps)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program was stopped with error")
