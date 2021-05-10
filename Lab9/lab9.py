from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

count = 0

def f(x, y):
    return ((1.3*(1-y**2))/(2*(x**2)+(y**2)+1))

def next_modified_y(x, y_prev, h):
    return y_prev + h*f(x+h/2, y_prev+h/2*f(x, y_prev))

def euler(h, a, b):
    y = 0
    y_k = []
    x_k = np.arange(a, b+h, h)
    for x in x_k:
        y_k.append(y)
        y = next_modified_y(x, y, h)
    return y_k, x_k
    
    
def runge_kutta(x, h, y):
    h2 = h / 2
    k1 = f(x, y)
    k2 = f(x + h2, y + h2 * k1)
    k3 = f(x + h2, y + h2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)    
    

def find_step(eps, a, b):
    h0 = eps ** (1/4)
    n = int((b - a) / h0)
    if n % 2 != 0 :
        n += 1
    while check_step(n, eps, a, b) == False:
        n = n*2
    n = n*4
    #print(n)
    return (b - a) / n


def check_step(n, eps, a, b):
    global count
    count += 1
    h = (b - a) / n 
    h2 = (b-a)/(2*n)
    y_k1 = []
    y_k2 = []
    y_01 = 0
    y_02 = 0
    x_k1 = np.arange(a+h, b+h, h) 
    x_k2 = np.arange(a, b+h, h2)  
    for x in x_k1:
        y_k1.append(y_01)
        y_01 = runge_kutta(x, h, y_01)  
    for x in x_k2:
        y_k2.append(y_02)
        y_02 = runge_kutta(x, h2, y_02)   
    for i in range(0, len(y_k1), 1):
        if i == 0:
            j = 0
        else:
            j = i+2
        if round(abs(y_k1[i]-y_k2[j]), 5) > eps:
            #print(round(abs(y_k1[i]-y_k2[j]), 5))
            if count >= 15:
                return True
            return False
    return True
    
    
def main():
    eps = 0.001
    a = 0
    b = 1
    h = find_step(eps, a, b)
    #print("Шаг интегрирования: ", h)
    y = 0
    y_k_e, x_k_e = euler(h, a, b)
    y_k_rk = []
    x_k_rk = np.arange(a+h, b+h, h)   
    for x in x_k_rk:
        y_k_rk.append(y)
        y = runge_kutta(x, h, y) 
    res_e = (odeint(f, y_k_e[0], x_k_e))
    res_rk = (odeint(f, y_k_rk[0], x_k_rk))
    plt.plot(x_k_e, y_k_e, x_k_rk, res_rk, x_k_rk, y_k_rk)
    plt.show()
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Program was stoped with error")
