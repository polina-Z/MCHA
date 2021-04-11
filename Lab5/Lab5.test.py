#!/usr/bin/env python3
import numpy as np


def main():
    A = np.array([
        [-3.0, 5.0, 7.0],
        [5.0, 4.0, -1.0],
        [7.0, -1.0, 0.0]])

    dim = len(A)
    eps = 0.0001
    np.set_printoptions(suppress = True, precision = 4, floatmode = "fixed")
    
    print('\nМатрица А:\n')
    printMatrix(A)
    diagonal_matryx, V, iterations = ortogonal_matrix(A.copy(), eps, dim)
    print('\nПреобразованная матрица А:\n')
    printMatrix(diagonal_matryx)
    print('\nМатрица V после {0} итераций:\n'.format(iterations))
    printMatrix(V)
    eig = np.linalg.eig(A) 
    eigVect = np.array([0.0, 0.0, 0.0])  
     
    for i in range(len(diagonal_matryx)):
	    eigVect[i] = diagonal_matryx[i][i]
	
    print("\nСобственные векторы методом Якоби(вращений):\n")
    printVect(eigVect)
    print('Собственные значения NumPy:\n')
    printVect(eig[0])
    print('\nСобственные векторы методом Якоби(вращений):\n')
    printMatrix(V)
    print('\nСобственные векторы NumPy:\n')
    printMatrix(eig[1])
    #print('\nПроверим правильность найденных собственных векторов:')
    #for i in range(len(V)):
        #print(f"{A.dot(np.transpose(V)[i])} = {diagonal_matryx[i][i] * np.transpose(V)[i]}")


def ortogonal_matrix(A, eps, dim):
    iterations = 0
    V_for_eiginvectors = np.identity(len(A))
    while(True):
        A_prev = A.copy()
        iterations += 1
        i, j = max_no_diagonal_element(A)

        if (A[i][i] == A[j][j]):
       	    p = np.pi / 4
        else:
            p = 2 * A[i][j] / (A[i][i] - A[j][j])
        co = np.cos(1/2 * np.arctan(p))
        si = np.sin(1/2 * np.arctan(p))
        V = np.eye(dim)
        V[i][i] = co
        V[i][j] = -si
        V[j][i] = si
        V[j][j] = co
        A = np.transpose(V).dot(A).dot(V)
        V_for_eiginvectors = V_for_eiginvectors.dot(V)
        sum_all = 0
        sum_diag = 0
        for row in range(len(A)):
            sum_diag += A[row, row] ** 2
            for col in range(len(A)):
                sum_all += A[row, col] ** 2
        if abs(sum_all - sum_diag) < eps:
            break
    print('\nКоличество итераций (при погрешности {0}): {1}'.format(eps, iterations))
    return(A, V_for_eiginvectors, iterations)


def max_no_diagonal_element(A):
    absolute_A = np.absolute(A)
    current_max = absolute_A[0][1]
    i, j = 0, 1
    for row in range(len(A)):
        for col in range(row + 1, len(A)):
            if absolute_A[row][col] > current_max:
                i, j, current_max = row, col, absolute_A[row][col]
    return (i,j)


def printMatrix(A):
    for row in range(len(A)):
        for col in range(len(A[row])):
            print('{:>8}'.format("%.4f" % A[row][col]), end = ' ')
        print()
        

def printVect(A):
    print("[", end='')
    for row in range(len(A)):
        print(" ", end = '')
        print("%.4f" % A[row], end='')
    print("]\n")
    
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program was stoped with error")
