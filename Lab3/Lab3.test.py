import numpy

numpy.random.seed(42)
EPS = 10.0 ** -5
Iters = 0


def input_polynomial():
    expr = numpy.poly1d([5.0, -6.0, 1.0])
    return expr


(f) = input_polynomial()


def sturm_seq(f):
    arr = []
    arr.append(f)
    arr.append(numpy.polyder(f))

    while True:
        fn = -numpy.polydiv(arr[-2], arr[-1])[1]
        if fn.order > 0 or abs(fn[0]) > 0.0:
            arr.append(fn)
        else:
            break
    return arr


def N(st_seq, x):
    if abs(f(x)) < EPS:
        raise ValueError("Number in N is a root")
    answer = 0
    for i in range(1, len(st_seq)):
        if st_seq[i](x) == 0.0:
            raise ValueError("SturmSeq[i] is zero")
        if st_seq[i - 1](x) * st_seq[i](x) < 0:
            answer += 1
    return answer


def calc_bounds(f, left, right):
    if (abs(f(left)) < EPS) or (abs(f(right)) < EPS):
        raise ValueError("Bounds contain root")
    if N(Sturm, left) - N(Sturm, right) == 0:
        return []
    if N(Sturm, left) - N(Sturm, right) > 1:
        while True:
            med = left + (right - left) / (1.5 + numpy.random.random())
            if abs(f(med)) > EPS:
                break
        return calc_bounds(f, left, med) + calc_bounds(f, med, right)
    if right - left < EPS:
        print("Warning: Bounds are too small")
    return [(left, right)]


Sturm = sturm_seq(f)
bounds = calc_bounds(f, -3, 3)


def half_division_method(left, right):
    global Iters
    Iters += 1
    med = (left + right) / 2
    if right - left < EPS:
        return med
    if f(left) * f(med) <= 0:
        return half_division_method(left, med)
    elif f(right) * f(med) <= 0:
        return half_division_method(med, right)
    else:
        raise RuntimeError("Something went wrong in binarySearch")


def chord_method(left, right):
    global Iters
    (x, old_x) = (left, right)
    while abs(x - old_x) > EPS:
        Iters += 1
        old_x = x
        x = left - f(left) * (right - left) / (f(right) - f(left))
        if f(left) * f(x) <= 0:
            right = x
        elif f(right) * f(x) <= 0:
            left = x
        else:
            raise RuntimeError("Something is wrong in chord method")
    return x


def newton_method(left, right):
    global Iters
    f_der = numpy.polyder(f)
    f_der2 = numpy.polyder(f, 2)
    if f(left) * f_der2(left) > 0:
        (old_x, x) = (right, left)
    elif f(right) * f_der2(right) > 0:
        (old_x, x) = (left, right)
    else:
        raise ValueError("Bad bounds in Newton method")
    while abs(x - old_x) > EPS:
        Iters += 1
        old_x = x
        x = x - f(x) / f_der(x)
    if x < left or x > right:
        raise RuntimeError("Something is wrong in newton method")
    return x


def wrapping_function(method):
    global Iters
    Iters = 0
    sum_iters = 0
    results = []
    iterations = []

    try:
        for i in range(len(bounds)):
            Iters = 0
            results.append(method(*bounds[i]))
            sum_iters += Iters
            iterations.append(Iters)
        min_number = results[0]
        Iters = iterations[0]
        k = 0
        for n in results:
            if n < min_number:
                min_number = n
                Iters = iterations[k]
            k += 1
        if min_number is not None:
            min_number = "{:.4f}".format(min_number)
        print(
            "Method: {} \t  {}  (total number of iterations: {}, for the minimum root: {})".format(
                method.__name__, min_number, sum_iters, Iters
            )
        )
        Iters = 0
    except Exception as ex:
        print(
            "ERROR: {} - in {} method (total number of iterations: {}б  for the minimum root: {})".format(
                ex, method.__name__, sum_iters, Iters
            )
        )

def test(method):
    global Iters
    for i in range(len(bounds)):
        Iters = 0
        try:
            str = method(*bounds[i])
            if  (not str is None):
                str = "{:.4f}".format(str)
            print("Method: {} \t  {}  (number of iterations: {})".format(method.__name__, str, Iters))
        except Exception as ex:
            print("ERROR: {} - in {} method (with {} iterations)".format(ex, method.__name__, Iters))


def main():
    print("f(x):\n", f)

    print("\nNumber of roots in the interval [-3, 3]:\t", N(Sturm, -3) - N(Sturm, 3), "\n")

    print("Bounds with roots:")
    print(bounds)

    numpy.set_printoptions(suppress=True, precision=5, floatmode="fixed")

    print()
    test(half_division_method)
    print()
    test(chord_method)
    print()
    test(newton_method)
    print()

    print("Checking with built-in function:")
    print(f.r)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program was stopped with error")