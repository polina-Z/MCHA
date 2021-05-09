import numpy as np


def my_func(x):
    return np.sinh(1 / x)


def first_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def second_diff(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


def main():
    a = 1
    b = 2
    x = (b + a) / 2
    h = 1 / 1000
    first_d = first_diff(my_func, x, h)
    second_d = second_diff(my_func, x, h)
    print(
        "Первая производная функции sinh(1/x) в точке x={}: {}".format(
            x, round(first_d, 5)
        )
    )
    print(
        "Вторая производная функции sinh(1/x) в точке x={}: {}".format(
            x, round(second_d, 5)
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program was stoped with error")
