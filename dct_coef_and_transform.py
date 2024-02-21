import time
from random import random

import matplotlib.pyplot as plt
import numpy as np

from utils import *


# 每一个uv对应的dct基，本质是一个变换矩阵
def plotdct2base(u, v, M, N, flag):
    A = u*np.pi/(2*M)
    B = v*np.pi/(2*N)
    row = np.arange(M)
    col = np.arange(N)

    row = np.cos((2*row+1)*A)
    col = np.cos((2*col+1)*B)

    dct2base = np.matmul(row.reshape(M, 1), col.reshape(1, N))
    dct2baseplot = None
    if flag:
        # 如果 A 是一个 m×n 矩阵，B 是一个 p×q 矩阵，那么它们的 Kronecker 积 C = np.kron(A, B) 将是一个 mp×nq 矩阵
        dct2baseplot = np.kron(dct2base, np.ones((32, 32)))
        # print("dct2base.shape,dct2baseplot.shape")
        # print(dct2base.shape, "x(32,32)", dct2baseplot.shape)
    return dct2base, dct2baseplot

# 每一个uv对应的idct基，本质也是一个变换矩阵


def plotidct2base(i, j, M, N, flag):
    C = np.ones((np.max([M, N]), 1))
    C[0] = np.sqrt(2)/2

    A = (2*i+1)*np.pi/(2*M)
    B = (2*j+1)*np.pi/(2*N)
    row = np.arange(M)
    col = np.arange(N)

    scale = 2*np.matmul(C.reshape(M, 1), C.reshape(1, N))/np.sqrt(M*N)

    row = np.cos(row*A)
    col = np.cos(col*B)
    idct2base = np.matmul(row.reshape(M, 1), col.reshape(1, N))*scale
    idct2baseplot = None
    if flag:
        idct2baseplot = np.kron(idct2base, np.ones((32, 32)))
    return idct2base, idct2baseplot

# 展示DCT系数图


def exp1():
    M = 4
    N = 4
    flag = 1
    plt.figure()
    for u in range(M):
        for v in range(N):
            dct2base, dct2baseplot = plotdct2base(u, v, M, N, flag)
            plt.subplot(M, N, u*N+v+1)
            plt.imshow(dct2baseplot, 'gray', vmin=0, vmax=1)  # 仅关系show
            plt.axis('off')
    plt.show()

# 展示二维DCT和两个一维DCT（改进）的区别


def exp2():
    # 固定
    # data = [[202, 205, 189, 188, 189, 175, 175, 175],
    #         [200, 203, 198, 188, 189, 182, 178, 175],
    #         [203, 200, 200, 195, 200, 187, 185, 175],
    #         [200, 200, 200, 200, 197, 187, 187, 187],
    #         [200, 205, 200, 200, 195, 188, 187, 175],
    #         [200, 200, 200, 200, 200, 190, 187, 175],
    #         [205, 200, 199, 200, 191, 187, 187, 175],
    #         [210, 200, 200, 200, 188, 185, 187, 186]]
    # data = np.array(data)

    # 随机
    data = np.random.randint(low=0, high=255, size=(8, 8))

    M = 8
    N = 8
    flag = 0

    # 2D dct  # 复杂度M^2N^2
    C = np.ones((np.max([M, N]), 1))
    C[0] = np.sqrt(2)/2
    scale = 2*np.matmul(C.reshape(M, 1), C.reshape(1, N))/np.sqrt(M*N)

    st1 = time.time()
    dct_2D_coef = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            dct2base, _ = plotdct2base(u, v, M, N, flag)
            dct_2D_coef[u, v] = (data*dct2base).sum()  # 一个变换之后，能量集中在一点
    dct_2D_coef = dct_2D_coef * scale

    # 2D idct
    data_rec_2D = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            idct2base, _ = plotidct2base(i, j, M, N, flag)
            # 一个逆变换，从所有坐标的能量中提取恰当分量，来还原
            data_rec_2D[i, j] = (dct_2D_coef*idct2base).sum()
    end1 = time.time()

    print("data=\n", data)
    print()

    print("The reconstructed version of original data for 2D DCT/IDCT Transform:")
    print(np.floor(data_rec_2D+0.5))
    print()
    print(f'Runtime is {end1-st1:.8f} s for {M} x {N} 2D-DCT/IDCT Transform')

    # 二维DCT转两个一维DCT，但本质还是向量乘法
    st2 = time.time()
    for _ in range(10000):  # 方便计时
        row = np.arange(M)
        col = 2*np.arange(N)+1
        T = np.sqrt(2/N)*np.cos(np.matmul(row.reshape(M, 1),
                                          col.reshape(1, N))*np.pi/(2*N))
        T[0, :] = 1/np.sqrt(N)

        dct_coef = np.matmul(np.matmul(T, data), T.T)
        data_rec_matrix = np.floor(
            np.matmul(np.matmul(T.T, dct_coef), T) + 0.5)
    end2 = time.time()

    print("The improved reconstructed version of original data for 2D DCT/IDCT Transform:")
    print(np.floor(data_rec_matrix+0.5))

    print()
    print(
        f'Runtime is {(end2-st2)/10000:.8f} s for {M} x {N} 2D-DCT/IDCT Transform')


def exp3():
    x = np.random.randint(low=0, high=255, size=[9, 9])
    print('original:\n', x)
    x, pad1, pad2 = padding(x, [4, 4])
    T = getT(4, 4)
    x_coef = blockproc(img=x, blksize=[4, 4], func=dct, T=T)
    x = blockproc(img=x_coef, blksize=[4, 4], func=idct, T=T)
    x = unpadding(x, pad1, pad2)
    print('DCT rec:\n', x)


if __name__ == '__main__':
    # exp1()
    # exp2()
    exp3()
