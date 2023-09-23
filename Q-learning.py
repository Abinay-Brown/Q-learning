import numpy as np
import control as ct


def LQR_gain():
    n, m = 3, 1;
    A = np.array([[-1.01887, 0.90506, -0.00215], [0.82225, -1.07741, -0.17555],[ 0, 0, -1]]);
    B = np.array([[0.0],[0.0],[1.0]]);
    M = np.eye(n, n);
    R = 0.1*np.eye(m, m);
    L = ct.lqr(A, B, M, R)
    return L



if  __name__ == "__main__":

    print(LQR_gain())