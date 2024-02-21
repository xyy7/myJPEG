import contextlib

from dct_coef_and_transform import *
from HuffmanCode.huffmancoding import *
from PIL import Image
from skimage.metrics import structural_similarity as cal_ssim
from utils import *

# Z字型
ZigZag = [
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63]

# 不同课本不同，但是理论上应该没有重复的才是正确的
# <Still Image and Video Compression with MATLAB>
dc_table = [
    "0",
    "10",
    "110",
    "1110",
    "11110",
    "111110",
    "11111110",
    "111111110",
    "1111111110",
    "11111111110",
    "111111111110",
    "1111111111110",
    "11111111111110",
    "111111111111110",
    "1111111111111110",
    "1111111111111111"
]


def get_lumaac():
    lumaac = []
    with open('F:\\szu18-onedrive\\OneDrive - email.szu.edu.cn\\课程\研一上-高级图像和视频压缩\\myJPEG\\myJPEGPlus\\lumaac.txt') as file:
        contents = file.readlines()
        for content in contents:
            if content == "\n":
                continue
            acs = content.split('\t')
            for ac in acs:
                if ac.strip('\n').strip('\t') == '1111000':  # 课本的码表有问题，两本都有
                    lumaac.append('11110000')
                    continue
                if ac.strip('\n').strip('\t') == '11111111000':
                    lumaac.append('111111110000')
                    continue
                lumaac.append(ac.strip('\n').strip('\t'))
    lumaac.append('1010')    # EOB
    lumaac.append('1111001')  # (15,0)
    return lumaac


def get_chromaac():
    chromaac = []
    with open('F:\\szu18-onedrive\\OneDrive - email.szu.edu.cn\\课程\研一上-高级图像和视频压缩\\myJPEG\\myJPEGPlus\\chromaac.txt') as file:
        contents = file.readlines()
        for content in contents:
            if content == "\n":
                continue
            # print(content)
            acs = content.split('\t')
            for ac in acs:
                chromaac.append(ac.strip('\n').strip('\t'))
    chromaac.append('00')  # EOB
    chromaac.append("1111111010")  # (15,0)1111111010
    return chromaac


lumaac = get_lumaac()
chromaac = get_chromaac()
print(len(lumaac), len(set(lumaac)))
print(len(chromaac), len(set(chromaac)))

# 查找錯誤的哈夫曼前綴
# for i in range(162):
#     for j in range(162):
#         if i == j:
#             continue
#         if lumaac[i].startswith(lumaac[j]):
#             print(i, j)  # 课本给的huffman前缀是有相同的 两本书都是有错误的
#             print(lumaac[i], lumaac[j])
#             print("error!!!")
# exit()


def write_symbol(symbol, bitout):
    size = 0
    for s in symbol:
        size += bitout.write(int(s))
    return size


def read_symbol(inp, table):
    hashtable = set(table)
    s = ''
    while True:
        s += str(inp.read_no_eof())
        if s in hashtable:
            for i, x in enumerate(table):
                if x == s:
                    return i, s  # size_category


def write_int(val, bit, bitout):
    size = 0
    for j in reversed(range(bit)):
        size += bitout.write((val >> j) & 1)
    return size


def read_int(n, inp):
    result = 0
    for _ in range(n):
        bit = inp.read_no_eof()
        result = (result << 1) | bit  # Big endian
    return result


def getsize(val):
    # 为什么使用<size,amplititude>? amplititude直接编码是因为赋值变化范围大，使用哈夫曼编码已经得不偿失。
    # return np.ceil(np.log2(np.abs(val)+1)+0.5)
    if val == 0:
        return 0
    return np.ceil(np.log2(np.abs(val)+0.5))


def getindex(runlength, dcsize):
    # 返回acindex
    if runlength == 0 and dcsize == 0:
        return 160
    if runlength == 15 and dcsize == 0:
        return 161
    index = runlength*10+dcsize-1
    return index


def dpcm(ac_coef, bitout, dctable):
    # 针对序对 (SIZE, AMPLITUDE) 中的SIZE，会采用哈夫曼编码，同时产生一张哈夫曼编码表。而AMPLITUDE则直接用二进制bit串表示。
    def diff(img):
        img = np.array(img, dtype='int')
        img[1:] = img[1:]-img[:-1]
        return img

    ac_coef = diff(ac_coef)
    size = 0
    for ac in ac_coef:
        acsize = int(getsize(ac))
        symbol = dctable[acsize]
        size += write_symbol(symbol, bitout)    # size固定哈夫曼表编码
        size += write_int(ac, acsize+1, bitout)  # amplitude直接编码
    # ac.write(256)  # 结束标识符（需要超过最大范围）  # 可以通过编码长宽来解决
    return size


def idpcm(bitin, table, numOfDC):
    # 可以通过记录一个数量来标记结束，但可能并非如此简单
    def reverse_diff(img):
        for i in range(1, img.shape[0]):
            img[i] = img[i]+img[i-1]
        return img

    dc_coef = []

    for i in range(numOfDC):
        size_category, s = read_symbol(bitin, table)

        dc = read_int(size_category+1, bitin)
        if dc >= 2**size_category:  # -6 26 (4+1) -4 12 (3+1)
            dc = -(2**(size_category+1)-dc)
        dc_coef.append(dc)

    dc_coef = reverse_diff(np.array(dc_coef))
    return dc_coef


def rle(dc_coef, bitout, table):
    # 针对序对 ((RunLength/SIZE), AMPLITUDE) 中的(RunLength/SIZE)，会采用哈夫曼编码，同时产生一张哈夫曼编码表。而AMPLITUDE则直接用二进制bit串表示。
    size = 0
    # with contextlib.closing(BitOutputStream(open(outfile, "wb"))) as bitout:
    num = 0
    for dc in dc_coef:
        if dc[1] == 0:
            dcsize = 0
        else:
            dcsize = int(getsize(dc[1]))
        runlength = dc[0]
        index = getindex(runlength, dcsize)
        symbol = table[index]
        # if num>15900:
        #     print('rle:',num,index,symbol)

        num += 1

        size += write_symbol(symbol, bitout)
        if index >= 160:  # 160 （0，0） 161 （15，0） 直接进行解析
            continue

        index = index+1
        if index % 10 == 10:
            size_category = 10
        else:
            size_category = index % 10

        size += write_int(dc[1], size_category+1, bitout)
    print(size)
    return size


def irle(bitin, table, numOfAC):
    dc_coef = []

    size = 0
    for i in range(numOfAC):
        index, s = read_symbol(bitin, table)  # 跟dpcm不同，需要解析一下index
        size += len(s)
        # print(size//8)
        # if i>15900:
        # print(i,index,s)
        if index == 160:
            # 遇到结束标志，后面的
            dc_coef.append([0, 0])
            continue
        if index == 161:
            dc_coef.append([15, 0])
            continue

        index = index+1
        if index % 10 == 10:
            size_category = 10
            dc1 = index // 10-1
        else:
            size_category = index % 10
            dc1 = index // 10

        dc2 = read_int(size_category+1, bitin)
        if dc2 >= 2**size_category:  # -6 26 (4+1) -4 12 (3+1)
            dc2 = -(2**(size_category+1)-dc2)
        dc_coef.append([dc1, dc2])
        size += size_category+1
    return dc_coef


def reverseZigZag(block, zigzag):
    new_block = np.zeros(block.shape)
    for i in range(64):
        new_block[i] = block[zigzag[i]]
    return np.reshape(new_block, [8, 8])


def dct_jpeg_entropy_plus(filename, new_filename, q):
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

    print(jpgQstepsC)
    print(jpgQstepsY)

    img = Image.open(filename)
    img_yuv = img.convert('YCbCr')

    img_yuv_offset = np.array(img_yuv, dtype='float') - 128
    img_yuv_rec = np.array(img)

    AREA = img_yuv_offset.shape[0]*img_yuv_offset.shape[1]
    size = 0
    for i in range(3):
        yuv, pad1, pad2 = padding(
            img_yuv_offset[:, :, i], blksize)
        m, n = yuv.shape
        M, N = blksize
        T = getT(M, N)

        # 压缩
        ac_coef = []
        dc_coef = []
        outbin = 'output.bin'
        outjpg = 'output.jpg'

        # 1. zigzag编码
        hdata = np.hsplit(yuv, n/N)  # 垂直分成高度度为8 的块
        for k in range(0, n//N):
            blockdata = np.vsplit(hdata[k], m/M)
            # 垂直分成高度为8的块后,在水平切成长度是8的块, 也就是8x8 的块
            for j in range(0, m//M):

                blockdata[j] = dct(T, blockdata[j])
                if i == 0:
                    blockdata[j] = np.floor(blockdata[j]/jpgQstepsY+0.5)
                else:
                    blockdata[j] = np.floor(blockdata[j]/jpgQstepsC+0.5)

                arr = [0]*64
                notnull_num = 0
                block = blockdata[j]
                for x in range(0, 64):
                    tmp = int(block[int(x/8)][x % 8])
                    arr[ZigZag[x]] = tmp
                    # 统计arr数组中有多少个非0元素
                    if tmp != 0 and x != 0:
                        notnull_num += 1
                # RLE编码
                # 标识连续0的个数
                dc_coef.append(arr[0])
                time = 0
                for x in range(1, 64):
                    if arr[x] == 0 and time < 15:
                        time += 1
                    else:
                        if arr[x] != 0:
                            notnull_num -= 1
                        # 如果0写入，那么这个0和前面15个0是有区别的。也就是说(15,0)代表16个0【这里应该由自己定义】
                        ac_coef.append([time, arr[x]])
                        time = 0
                        if notnull_num == 0:
                            ac_coef.append([0, 0])  # 作为终止符标志
                            break

                hdata[k] = np.vstack(blockdata)  # back
            x_h = np.hstack(hdata)

        # 2. 对直流系数使用dpcm，对交流系数使用rle
        chl_size = 0
        with contextlib.closing(BitOutputStream(open(outbin, "wb"))) as bitout:
            chl_size += dpcm(dc_coef, bitout, dc_table)
            if i == 0:
                sizerle = rle(ac_coef, bitout, lumaac)
            else:
                sizerle = rle(ac_coef, bitout, chromaac)
            # print('sizerle',sizerle)   # 10756
            # exit()
            chl_size += sizerle
            print(f'original:{AREA} bytes,compressed:{chl_size} bytes')
            print('compress ratio:', AREA/chl_size)
        size += chl_size
        print('len:', len(dc_coef), len(ac_coef))

        # 3.使用idpcm解码出直接系数，使用irle解码出交流系数
        with open(outbin, "rb") as inp:
            bitin = BitInputStream(inp)
            dc_coef1 = idpcm(bitin, dc_table, len(dc_coef))
            dc_coef = list(dc_coef1)

            # s = ''
            # for i in range(10756*8): # 15972
            #     s=bitin.read_no_eof()
            #     print(i//8)

            if i == 0:
                # 应该记录结束标志，本代码主要为了展示dpcm和rle，故header等细节没有考虑
                ac_coef1 = irle(bitin, lumaac, len(ac_coef))
            else:
                ac_coef1 = irle(bitin, chromaac, len(ac_coef))
            bitin.close()

        ac_coef = ac_coef1

        # exit()

        ac_coef = ac_coef[::-1]  # 因为下面使用pop
        dc_coef = dc_coef[::-1]

        # 4.将直流系数和交流系数，通过reverseZigZag复原
        x_h = np.zeros(x_h.shape)   # 理论上应该写进头文件，这里主要为了演示dct，所有省略
        hdata = np.hsplit(x_h, n/N)  # 垂直分成高度度为8 的块
        for k in range(0, n//N):
            blockdata = np.vsplit(hdata[k], m/M)
            # 垂直分成高度为8的块后,在水平切成长度是8的块, 也就是8x8 的块
            for j in range(0, m//M):
                block = np.zeros(64)
                block[0] = dc_coef.pop()

                numofac = 0
                while True:
                    ac = ac_coef.pop()
                    # 如果最后的数字都是0
                    if ac[0] == 0 and ac[1] == 0:
                        break
                    numofac += ac[0]+1
                    block[numofac] = ac[1]
                    # 如果最后一个数字非0
                    if numofac == 63:
                        break

                # blockdata[j] = np.reshape(block,[8,8])
                blockdata[j] = reverseZigZag(block, ZigZag)
                if i == 0:
                    blockdata[j] = blockdata[j]*jpgQstepsY
                else:
                    blockdata[j] = blockdata[j]*jpgQstepsC
                blockdata[j] = idct(T, blockdata[j])

            hdata[k] = np.vstack(blockdata)  # back
        y_h = np.hstack(hdata)

        img_yuv_rec[:, :, i] = unpadding(y_h, pad1, pad2)+128

    img_rec = Image.fromarray(
        np.array(img_yuv_rec, dtype='uint8'), 'YCbCr').convert('RGB')

    Image.fromarray(np.array(img_rec, dtype='uint8')).save(new_filename)
    img_rec = np.array(img_rec, dtype='uint8')
    img = np.array(img, dtype='uint8')
    ssim = cal_ssim(img_rec, img, data_range=255, multichannel=True)
    mse = cal_mse(img, img_rec)
    psnr = cal_psnr(mse)

    H, W, C = img.shape
    original = H*W
    numel = original * C
    bpp = size*8/original

    print()
    print(filename)
    print(f'quality:{q}')
    # print(f'encoded time:{compress_time:.4f} s')
    # print(f'decoded time:{decompress_time:.4f} s')
    print(f'original size:{numel} bytes')
    print(f'compressed size:{size} bytes')
    print(f'compressed ratio:{numel/size:.4f} ')
    print(f'bpp:{bpp:.4f}, SSIM:{ssim:.4f},MSE:{mse:.4f},PSNR:{psnr:.4f} dB')
    print()

    return ssim, bpp, psnr
