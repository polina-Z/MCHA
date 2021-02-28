import numpy as np


def simple_iteration_method(A, b, init_vector=None):
    B = np.eye(A.shape[0]) - A / np.diag(A).reshape(-1, 1)
    if min(np.linalg.norm(B, ord=norm) for norm in (np.inf, 1, "fro")) >= 1:
        raise ValueError("Simple iteration method cannot be performed")
    c = b / np.diag(A).reshape(-1, 1)
    if init_vector is None:
        c.dtype = float
        xk = c.copy()
    else:
        init_vector.dtype = float
        xk = init_vector.copy()
    iteration_number_method1 = 2
    while True:
        xk_1 = xk.copy()
        xk = B @ xk_1 + c
        eps = np.linalg.norm(B) * np.linalg.norm(xk - xk_1) / (1 - np.linalg.norm(B))
        if eps < 1e-4:
            break
        xk_1 = xk.copy()
        iteration_number_method1 += 1
    print(iteration_number_method1)
    print()
    return xk


def main():
    k = 10

    C = np.array(
        [
            [0.01, 0, -0.02, 0, 0],
            [0.01, 0.01, -0.02, 0, 0],
            [0, 0.01, 0.01, 0, -0.02],
            [0, 0, 0.01, 0.01, 0],
            [0, 0, 0, 0.01, 0.01],
        ],
        dtype=float,
    )

    D = np.array(
        [
            [1.33, 0.21, 0.17, 0.12, -0.13],
            [-0.13, -1.33, 0.11, 0.17, 0.12],
            [0.12, -0.13, -1.33, 0.11, 0.17],
            [0.17, 0.12, -0.13, -1.33, 0.11],
            [0.11, 0.67, 0.12, -0.13, -1.33],
        ],
        dtype=float,
    )

    b = np.array([[1.2], [2.2], [4.0], [0.0], [-1.2]], dtype=float)

    A = np.add(np.multiply(C, k), D)
    print("Matrix A:")
    for row in A:
        print(" ".join([str(round(elem, 8)) for elem in row]))
    for diag in np.diag(A).reshape(-1, 1):
        if diag == 0.0:
            print("Simple iteration method cannot be performed")
            return 0
    print()
    _true_x = (np.linalg.inv(A) @ b).reshape(-1)
    print("Number of iteration steps:")
    print(f"{'x =':<1} {simple_iteration_method(A, b).reshape(-1)}{'T'}")


if __name__ == "__main__":
    main()
