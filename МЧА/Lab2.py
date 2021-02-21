import numpy as np

C = np.array([[0.2, 0, 0.2, 0, 0],
    [0, 0.2, 0, 0.2, 0],
    [0.2, 0, 0.2, 0, 0.2],
    [0, 0.2, 0, 0.2, 0],
    [0, 0, 0.2, 0, 0.2]])

D = np.array([[2.33, 0.81, 0.67, 0.92, -0.53],
    [-0.53, 2.33, 0.81, 0.67, 0.92],
    [0.92, -0.53, 2.33, 0.81, 0.67],
    [0.67, 0.92, -0.53, 2.33, 0.81],
    [0.81, 0.67, 0.92, -0.53, 2.33]])
    
b = np.array([4.2, 4.2, 4.2, 4.2, 4.2]).reshape(-1,1)
k = 10

def matrixA():

    extendedA = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]])

    for i in range(len(C)):
        for j in range(len(C[i])):
            extendedA[i][j] = k * C[i][j] + D[i][j]
    extendedA = np.concatenate([extendedA,b],axis = 1)
    return extendedA
    
def systemIsCompatible(matrixExt):
    for i in range(len(matrixExt)):
        if not matrixExt[i][i]:
            return True
    return False
    
def toFixed(numObj, digits = 0):
    return f"{numObj:.{digits}f}" 
    
def maxInColumn(matrixExt, column):
    max_element = matrixExt[column][column]
    max_column = column
    for i in range(column + 1, len(matrixExt)):
        if abs(matrixExt[i][column]) > abs(max_element):
            max_element = matrixExt[i][column]
            max_column = i
    if max_column != column:
        matrixExt[column], matrixExt[max_column] = matrixExt[max_column], matrixExt[column]

def GaussMethod(matrixExt):
    n = len(matrixExt)         
    for j in range(n - 1):
        maxInColumn(matrixExt, j)
        for i in range(j + 1, n):
            if matrixExt[j][j] == 0:
                continue
            div = matrixExt[i][j] / matrixExt[j][j]
            matrixExt[i][-1] -= div * matrixExt[j][-1]
            for l in range(j, n):
                matrixExt[i][l] -= div * matrixExt[j][l]

    if systemIsCompatible(matrixExt):
        print('The system has infinite number of answers')
        print()
        return

    x = [0 for i in range(n)]
    for j in range(n - 1, -1, -1):
        x[j] = (matrixExt[j][-1] - sum([matrixExt[j][l] * x[l] for l in range(j + 1, n)])) / matrixExt[j][j]

    print(x)
    for i in range(len(x)):
        a = str(toFixed(x[i],4))
        print("x{} = {}".format(i+1,a), end = ' ')
    print()
       

def main():
    extendedA = matrixA()
    for row in extendedA:
        print(row)  
    GaussMethod(extendedA)
if __name__ == "__main__":
    main()
