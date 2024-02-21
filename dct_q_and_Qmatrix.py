import time

import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as cal_ssim

from utils import *


def dct_entropy_demo(filename, blksize, q):

    print(f'The block size of DCT is {blksize[0]}')
    print(f'The step size of midtread quantizer is {q}')

    img = Image.open(filename)
    img_yuv = img.convert('YCbCr')

    img_yuv_offset = np.array(img_yuv, dtype='float') - 128
    img_yuv_rec = np.array(img)
    st = time.time()
    T = getT(blksize[0], blksize[1])
    # TODO:多线程版本
    for i in range(3):
        img_yuv_offset_pad, pad1, pad2 = padding(
            img_yuv_offset[:, :, i], blksize)
        x = blockproc(img_yuv_offset_pad, blksize, dct, T)

        # 计算图像的直方图的熵
        x_h = np.floor(x/q+0.5)  # 四舍五入量化
        [Height, Width] = x_h.shape
        hist, bins = np.histogram(x_h, bins=256, range=(0, 255))
        p = hist/(Height*Width)
        entropy = (-p*np.log2(p+1e-08)).sum()
        print(f'Entropy of input image for channel {i} = {entropy}')

        # y_h = x*q # 没有量化
        y_h = x_h*q
        img_yuv_rec[:, :, i] = unpadding(
            blockproc(y_h, blksize, idct, T), pad1, pad2)+128
    end = time.time()
    img_rec = Image.fromarray(
        np.array(img_yuv_rec, dtype='uint8'), 'YCbCr').convert('RGB')
    # img_rec = Image.fromarray(
    #     np.array(img_yuv, dtype='uint8'), 'YCbCr').convert('RGB')

    img = np.array(img)
    img_rec = np.array(img_rec)

    Image.fromarray(np.array(img_rec, dtype='uint8')).save('rec_lenna.png')
    ssim = cal_ssim(img_rec, img, data_range=255, multichannel=True)
    # ssim=calculate_ssim(img,img_rec)
    mse = cal_mse(img, img_rec)
    psnr = cal_psnr(mse)

    print(
        f'SSIM:{ssim:.4f},MSE:{mse:.4f},PSNR:{psnr:.4f} dB, time cost: {end-st:.6f}s\n')

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(img)  # 仅关系show
    plt.title('original image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_rec)  # 仅关系show
    plt.title('reconstructed image')
    plt.axis('off')

    plt.show()


def dct_jpeg_entropy_demo(filename, q, entropy_coder=None):
    blksize = [8, 8]
    print(f'The block size of DCT is {blksize[0]}')
    print(f'The step size of midtread quantizer is {q}')

    jpgQstepsY = [[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]]
    jpgQstepsY = np.array(jpgQstepsY)*q
    jpgQstepsC = [[17, 18, 24, 47, 66, 99, 99, 99],
                  [18, 21, 26, 66, 99, 99, 99, 99],
                  [24, 26, 56, 99, 99, 99, 99, 99],
                  [47, 66, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99],
                  [99, 99, 99, 99, 99, 99, 99, 99]]
    jpgQstepsC = np.array(jpgQstepsC)*q

    img = Image.open(filename)
    img_yuv = img.convert('YCbCr')

    img_yuv_offset = np.array(img_yuv, dtype='float') - 128
    img_yuv_rec = np.array(img)

    for i in range(3):
        img_yuv_offset_pad, pad1, pad2 = padding(
            img_yuv_offset[:, :, i], blksize)
        x = blockproc(img_yuv_offset_pad, blksize, dct)

        # x_h = np.floor(x/q+0.5)
        # TODO: 使用repeat可以进行加速
        if i > 0:
            x_h = np.floor(blockproc(x, blksize, div(jpgQstepsC))+0.5)
        else:
            x_h = np.floor(blockproc(x, blksize, div(jpgQstepsY))+0.5)

        [Height, Width] = x_h.shape
        hist, bins = np.histogram(x_h, bins=int(
            x_h.max()-x_h.min()+1), range=(x_h.min(), x_h.max()))
        p = hist/(Height*Width)
        entropy = (-p*np.log2(p+1e-08)).sum()
        print(f'Entropy of input image for channel {i} = {entropy}')

        # y_h = x_h*q
        if i > 0:
            y_h = blockproc(x_h, blksize, mul(jpgQstepsC))
            # y_h = blockproc(x_h, blksize, mul(jpgQstepsC)) + 0.5
        else:
            y_h = blockproc(x_h, blksize, mul(jpgQstepsY))
            # y_h = blockproc(x_h, blksize, mul(jpgQstepsY)) + 0.5

        img_yuv_rec[:, :, i] = unpadding(
            blockproc(y_h, blksize, idct), pad1, pad2)+128

    img_rec = Image.fromarray(
        np.array(img_yuv_rec, dtype='uint8'), 'YCbCr').convert('RGB')

    img = np.array(img)
    img_rec = np.array(img_rec)

    Image.fromarray(np.array(img_rec, dtype='uint8')).save('rec_lenna.png')
    ssim = cal_ssim(img_rec, img, data_range=255, multichannel=True)
    # ssim=calculate_ssim(img,img_rec)
    mse = cal_mse(img, img_rec)
    psnr = cal_psnr(mse)

    print(f'SSIM:{ssim:.4f},MSE:{mse:.4f},PSNR:{psnr:.4f} dB')

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(img)  # 仅关系show
    plt.title('original image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_rec)  # 仅关系show
    plt.title('reconstructed image')
    plt.axis('off')

    plt.show()


if __name__ == '__main__':

    q = 50
    filename = 'Lenna.png'
    # dct_entropy_demo(filename, [4, 4], q)  # exp3
    # dct_entropy_demo(filename, [8, 8], q)  # exp3
    # dct_entropy_demo(filename, [16, 16], q)  # exp3
    # dct_entropy_demo(filename, [32, 32], q)  # exp3
    # dct_entropy_demo(filename, [64, 64], q)  # exp3

    dct_jpeg_entropy_demo(filename, q=3)  # exp4
