import numpy as np


def my_func(x):
    return np.sinh(1 / x)


def integral_rectangle(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a, b, n + 1)
    x_x = []
    for i in range(1, len(x), 1):
        x_x.append((x[i]+x[i-1])/2)
    sum = 0
    for i in range(1, len(x_x)+1):
        sum += f(x_x[i-1])
    return sum*h
    

def integral_trapezoid(f, a, b, n):
    h = (b-a)/n
    sum = h*(f(a)+f(b))/2 
    x = a+h
    while x<=b-h:
        sum += h*f(x)
        x += h
    return sum
    

def integral_simpson(f, a, b, n):
    h = (b-a)/(2*n)
    tmp_sum = float(f(a)) +float(f(b))
    for step in range(1, 2 * n):
        if step % 2 != 0:
            tmp_sum += 4 * float(f(a + step * h))
        else:
            tmp_sum += 2 * float(f(a + step * h))
    return tmp_sum * h / 3
 
    
    
def main():
    a = 1
    b = 2
    print(
        "Интеграл функции sinh(1/x) на интервале [{},{}] методом прямоугольников: {}".format(
            a, b, round(integral_rectangle(my_func, a, b, 1000), 6)
        )
    )
    
    print(
        "Интеграл функции sinh(1/x) на интервале [{},{}] методом трапеций: {}".format(
            a, b, round(integral_trapezoid(my_func, a, b, 1000), 6)
        )
    )

    print(
        "Интеграл функции sinh(1/x) на интервале [{},{}] методом Симпсона: {}".format(
            a, b, round(integral_simpson(my_func, a, b, 32), 6)
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program was stoped with error")
