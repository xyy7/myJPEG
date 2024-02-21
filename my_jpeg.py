import os
from pathlib import Path

from PIL import Image
from skimage.metrics import structural_similarity as cal_ssim

# from ArithmeticCode.myArithEntropyCoder import Image2bin
from dct_coef_and_transform import *
from dct_q_and_Qmatrix import *
from HuffmanCode.myHuffmanEntropyCoder import Image2bin
from myJPEGPlus.my_jpeg_plus import dct_jpeg_entropy_plus

Compresser = Image2bin()

# 添加了哈夫曼编码


def dct_jpeg_entropy_demo(filename, new_filename, q):
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

    compress_time = 0
    decompress_time = 0
    size = 0
    T = getT(blksize[0], blksize[1])
    # TODO: 可以concat q之后，去除for循环
    for i in range(3):
        img_yuv_offset_pad, pad1, pad2 = padding(
            img_yuv_offset[:, :, i], blksize)
        x = blockproc(img_yuv_offset_pad, blksize, dct, T)

        #x_h = np.floor(x/q+0.5)
        if i > 0:
            x_h = np.floor(blockproc(x, blksize, div(jpgQstepsC))+0.5)
        else:
            x_h = np.floor(blockproc(x, blksize, div(jpgQstepsY))+0.5)

        # 压缩
        s, ct = Compresser.main_compress(np.array(x_h, dtype='int'))
        compress_time += ct
        size += s
        # 解压缩
        dt, x_h = Compresser.main_decompress()
        decompress_time += dt
        x_h = np.squeeze(x_h)

        [Height, Width] = x_h.shape
        hist, bins = np.histogram(x_h, bins=int(
            x_h.max()-x_h.min()+1), range=(x_h.min(), x_h.max()))
        p = hist/(Height*Width)
        entropy = (-p*np.log2(p+1e-08)).sum()
        print(f'Entropy of input image for channel {i} = {entropy}')

        # y_h = x_h*q
        if i > 0:
            y_h = blockproc(x_h, blksize, mul(jpgQstepsC), T)
        else:
            y_h = blockproc(x_h, blksize, mul(jpgQstepsY), T)

        img_yuv_rec[:, :, i] = unpadding(
            blockproc(y_h, blksize, idct, T), pad1, pad2)+128

    img_rec = Image.fromarray(
        np.array(img_yuv_rec, dtype='uint8'), 'YCbCr').convert('RGB')

    img = np.array(img)
    img_rec = np.array(img_rec)

    print(img.shape, img_rec.shape)
    Image.fromarray(np.array(img_rec, dtype='uint8')).save(new_filename)
    ssim = cal_ssim(img_rec, img, data_range=255, multichannel=True)
    # ssim=calculate_ssim(img,img_rec)
    mse = cal_mse(img, img_rec)
    psnr = cal_psnr(mse)

    H, W, C = img.shape
    original = H*W
    numel = original * C
    bpp = size*8/original

    print()
    print(filename)
    print(f'quality:{q}')
    print(f'encoded time:{compress_time:.4f} s')
    print(f'decoded time:{decompress_time:.4f} s')
    print(f'original size:{numel} bytes')
    print(f'compressed size:{size} bytes')
    print(f'compressed ratio:{numel/size:.4f} ')
    print(f'bpp:{bpp:.4f}, SSIM:{ssim:.4f},MSE:{mse:.4f},PSNR:{psnr:.4f} dB')
    print()

    return ssim, bpp, psnr


def test_myJPEG(testdir, qfactor=[1, 2, 4, 6], func=dct_jpeg_entropy_demo):
    bpp = []
    psnr = []
    ssim = []
    rec_dir = 'rec_'+testdir
    os.makedirs(rec_dir, exist_ok=True)
    for q in qfactor:
        bppq = []
        psnrq = []
        ssimq = []
        for _, _, filenames in os.walk(testdir):
            for filename in filenames:
                ssim1, bpp1, psnr1 = func(f'{testdir}/{filename}',
                                          f'{rec_dir}/myJPEG_{q}_{filename}', q=q)
                bppq.append(bpp1)
                psnrq.append(psnr1)
                ssimq.append(ssim1)
                # exit()

        bpp.append(np.array(bppq).mean())
        psnr.append(np.array(psnrq).mean())
        ssim.append(np.array(ssimq).mean())
    return bpp, psnr, ssim


def test_PIL_jpg(testdir, qfactor=[20, 40, 60, 80]):
    bpp_PIL = []
    psnr_PIL = []
    ssim_PIL = []
    rec_dir = 'rec_'+testdir
    os.makedirs(rec_dir, exist_ok=True)

    for q in qfactor:
        bppq = []
        psnrq = []
        ssimq = []
        for _, _, filenames in os.walk(testdir):
            for filename in filenames:
                rec_path = f'{rec_dir}/PIL_{Path(filename).stem}_{q}.jpg'
                img = np.array(Image.open(f'{testdir}/{filename}'))
                H, W, C = img.shape
                numel = H*W
                rec_img = Image.fromarray(img).save(
                    rec_path, format='jpeg', quality=q)
                rec_img = np.array(Image.open(rec_path))
                bpp = os.path.getsize(rec_path)*8/numel
                psnr = cal_psnr(((rec_img-img)**2).mean())
                ssim = cal_ssim(rec_img, img, multichannel=True)
                bppq.append(bpp)  # bytes
                psnrq.append(psnr)
                ssimq.append(ssim)
                print(f'bpp:{bpp:.4f}, SSIM:{ssim:.4f}, PSNR:{psnr:.4f} dB')
                # exit()

        bpp_PIL.append(np.array(bppq).mean())
        psnr_PIL.append(np.array(psnrq).mean())
        ssim_PIL.append(np.array(ssimq).mean())
    return bpp_PIL, psnr_PIL, ssim_PIL


if __name__ == '__main__':

    testdir = 'Kodak_12'
    # func = dct_jpeg_entropy_plus
    func = dct_jpeg_entropy_demo
    bpp, psnr, ssim = test_myJPEG(testdir, func=func)
    bpp_PIL, psnr_PIL, ssim_PIL = test_PIL_jpg(testdir)

    print('bpp:\n', 'proposed:', bpp, '\n jpeg:', bpp_PIL)
    print("psnr:\n", 'proposed:', psnr, '\n jpeg:', psnr_PIL)
    print("ssim:\n", 'proposed:', ssim, '\n jpeg:', ssim_PIL)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('PSNR')
    # plt.grid('off')
    plt.xlabel('bpp')
    plt.ylabel('PSRN (dB')
    plt.plot(bpp, psnr, c='red', marker='+')
    plt.plot(bpp_PIL, psnr_PIL, c='blue', marker='x')
    plt.legend(['proposed', 'jpeg'])

    plt.subplot(1, 2, 2)
    plt.title('SSIM')
    # plt.grid('off')
    plt.xlabel('bpp')
    plt.ylabel('SSIM ')
    plt.plot(bpp, ssim, c='red', marker='+')
    plt.plot(bpp_PIL, ssim_PIL, c='blue', marker='x')
    plt.legend(['proposed', 'jpeg'])

    plt.show()
