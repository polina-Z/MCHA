import numpy


global EPS
global ITERATIONS


def first_equ(x, y):
    return x*y-y+1


def second_equ(x, y):
    return 5*x-y**2+2


def x_equ(y):
    return (y-1)/y


def y_equ(x):
    return numpy.sqrt(5*x+2)


def jacobi_matrix(x, y):
    return numpy.array([
        [y, x-1],
        [5, -2*y]])


def simple_iteration_method(x_0, y_0):
    global ITERATIONS
    ITERATIONS = 0
    (x, y) = (x_0, y_0)
    while True:
        ITERATIONS += 1
        old_x = x
        old_y = y
        x = x_equ(y)
        y = y_equ(x)
        if not (numpy.isfinite(x) and numpy.isfinite(y)):
            raise RuntimeError("Sequence {x} is divergent")
        if max(abs(x - old_x), abs(y - old_y)) < EPS:
            return x, y


def newton_method(x_0, y_0):
    global ITERATIONS
    ITERATIONS = 0
    (x, y) = (x_0, y_0)
    while True:
        ITERATIONS += 1
        j = jacobi_matrix(x, y)
        f = numpy.array([[first_equ(x, y)], [second_equ(x, y)]])
        x_x0 = numpy.linalg.solve(j, -f)
        x += x_x0[0][0]
        y += x_x0[1][0]
        if not (numpy.isfinite(x) and numpy.isfinite(y)):
            raise RuntimeError("Sequence {x} is divergent")
        if numpy.linalg.norm(x_x0) < EPS:
            return x, y


def wrapping_function(method, x_0, y_0):
    global ITERATIONS
    ITERATIONS = 0
    try:
        (x, y) = method(x_0, y_0)
        print(
            "Method: {} \t  (x, y) = ({:.4f}, {:.4f})  (number of iterations: {})".format(
                method.__name__, x, y, ITERATIONS))
    except Exception as ex:
        print("ERROR: {} - in {} (number of iterations: {}})".format(ex, method.__name__, ITERATIONS))
    print()


def main():
    global EPS
    EPS = 10.0 ** -5
    global ITERATIONS
    ITERATIONS = 0
    x_0 = 0.5
    y_0 = 2.0
    print("Initialization values =", (x_0, y_0))
    print()
    wrapping_function(simple_iteration_method, x_0, y_0)
    wrapping_function(newton_method, x_0, y_0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program was stopped with error")