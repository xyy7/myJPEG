import numpy as np
from skimage.metrics import structural_similarity as cal_ssim


def dct(T, data):
    # dct_coef
    return np.matmul(np.matmul(T, data), T.T)  # 单独使用的话，需要四舍五入，否则跟q一起四舍五入


def idct(T, dct_coef):
    # data_rec_matrix
    return np.floor(np.matmul(np.matmul(T.T, dct_coef), T) + 0.5)


def padding(img, blksize):
    m, n = img.shape
    M, N = blksize
    pad1 = M-m % M
    pad2 = N-n % N
    img = np.pad(img, ((0, pad1), (0, pad2)),
                 'constant', constant_values=(0, 0))
    return img, pad1, pad2


def unpadding(img, pad1, pad2):
    return img[:-pad1, :-pad2]


def getT(M=8, N=8):
    row = np.arange(M)
    col = 2*np.arange(N)+1
    T = np.sqrt(2/N)*np.cos(np.matmul(row.reshape(M, 1),
                                      col.reshape(1, N))*np.pi/(2*N))
    T[0, :] = 1/np.sqrt(N)
    return T


def blockproc(img, blksize, func, T=None):
    # 切分成4*4大小，然后进行dct变换，关键看如何连接
    m, n = img.shape
    M, N = blksize
    if T is None:
        T = getT(M, N)

    # TODO:可以通过repreat来替代for循环
    # print("img.shape(m, n), blksize(M, N)", m, n, M, N)
    hdata = np.hsplit(img, n//N)  # 垂直分成高度度为8 的块
    for i in range(0, n//N):
        blockdata = np.vsplit(hdata[i], m//M)
        # 垂直分成高度为8的块后,在水平切成长度是8的块, 也就是8x8 的块
        for j in range(0, m//M):
            if func == dct or func == idct:
                blockdata[j] = func(T, blockdata[j])
            else:
                blockdata[j] = func(blockdata[j])
        hdata[i] = np.vstack(blockdata)  # back
    img_coef = np.hstack(hdata)
    # print("img_coef.shape", img_coef.shape)
    return img_coef

# from 课件


def cal_mse(img, img_rec):
    return ((img-img_rec)**2).mean()

# from 课件


def cal_psnr(mse):
    return 10*np.log10(255**2/mse)


def mul(A):
    def multi(B):
        return A*B
    return multi


def div(A):
    def divi(B):
        return B/A
    return divi  # 返回函数指针
