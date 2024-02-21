import contextlib
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .huffmancoding import *
from utils import *

result_dir = 'hf_compress/'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


class Image2bin:
    """
    1. 支持差分
    2. 支持输入文件或者array
    3. 使用huffman进行和编码
    """

    def __init__(self, mode='V', file=None) -> None:
        if file is None:
            self.fullname = None
            self.filename = None
            self.type = 0  # array
        else:
            # mode表示是否采用差分编码
            self.initinize(mode=mode, img_src=file)

    def initinize(self, mode=None, img_src=None):
        if type(img_src) is str:  # file
            self.fullname = img_src
            self.filename = Path(img_src).stem
            self.img = np.array(Image.open(img_src).convert('YCbCr'))
            self.type = 1  # file
        else:
            # type == array
            self.fullname = None
            self.filename = None
            self.type = 0  # array
            self.img = img_src

        self.mode = mode
        if mode == 'V':
            self.img = self.difference_imgV(self.img)
        elif mode == 'H':
            self.img = self.difference_imgH(self.img)
        else:
            pass

        # 改成更加通用
        self.min = self.img.min()               # 需要进行记录
        self.max = self.img.max()
        self.interval = self.max - self.min+1     # 需要进行记录
        self.num = int(self.interval+1)
        # print('min,max,interval,num',self.min,self.max,self.interval,self.num)

        self.max_depth = int(np.ceil(np.log2(self.num+1)))
        self.img = self.img-self.min

        if len(self.img.shape) == 2:
            self.img = self.img[:, :, None]

        self.hist = list(self.get_hist()+1)  # 仅需要保证相对大小关系
        self.hist = [int(x) for x in self.hist]

    def difference_imgV(self, img):
        img = np.array(img, dtype='int')
        img[1:] = img[1:]-img[:-1]
        return img

    def reverse_dV(self, img):
        for i in range(1, img.shape[0]):
            img[i] = img[i]+img[i-1]
        return img

    def reverse_dH(self, img):
        for i in range(1, img.shape[1]):
            img[:, i] = img[:, i]+img[:, i-1]
        return img

    def difference_imgH(self, img):
        img = np.array(img, dtype='int')
        img[:, 1:] = img[:, 1:]-img[:, :-1]
        return img

    def get_hist(self):
        x, y, c = self.img.shape
        hist = np.zeros(self.num)
        for i in range(x):
            for j in range(y):
                for k in range(c):
                    hist[self.img[i, j, k]] += 1
        return hist

    def get_hist_single(self, img):
        hist, _ = np.histogram(img, bins=(
            img.max()-img.min()+1), range=(img.min(), img.max()))
        return hist

    def get_frequencies(self):
        return FrequencyTable(self.hist+[1])

    def write_code_len_table(self, bitout, canoncode):
        num_sym = canoncode.get_symbol_limit()
        # print("table length:",num_sym)
        size = 0
        for i in range(num_sym):
            val = canoncode.get_code_length(i)
            # For this file format, we only support codes up to 255 bits long
            if val >= num_sym:
                raise ValueError("The code for a symbol is too long")

            # Write value as 8 bits in big endian
            for j in reversed(range(self.max_depth)):
                size += bitout.write((val >> j) & 1)

        # print(f"table size: {size} bytes")
        return size

    def writeHeader(self, bitout):
        # print('img shape:',self.img.shape)
        # min:2B interval:2B
        # h 2B,w 2B,c 2b
        size = 0

        for j in reversed(range(16)):
            size += bitout.write((self.min >> j) & 1)

        for j in reversed(range(16)):
            size += bitout.write((self.interval >> j) & 1)

        if self.mode == 'V':
            mode = 1
        elif self.mode == 'M':
            mode = 2
        else:
            mode = 0
        for j in reversed(range(2)):
            size += bitout.write((mode >> j) & 1)

        h, w, c = self.img.shape
        for j in reversed(range(16)):
            size += bitout.write((h >> j) & 1)

        for j in reversed(range(16)):
            size += bitout.write((w >> j) & 1)

        for j in reversed(range(2)):
            size += bitout.write((c >> j) & 1)

        # print('min:2B,interval:2B,H:2B,W:2B,C:2bit')
        return size

    def read_code_len_table(self, bitin):
        def read_int(n):
            result = 0
            for _ in range(n):
                result = (result << 1) | bitin.read_no_eof()  # Big endian
            return result  # return depth

        codelengths = [read_int(self.max_depth)
                       for _ in range(len(self.hist)+1)]
        # print("len(codelenth)",len(codelengths))
        return CanonicalCode(codelengths=codelengths)

    def readHeader(self, bitin):
        def read_int(n):
            result = 0
            for _ in range(n):
                result = (result << 1) | bitin.read_no_eof()  # Big endian
            return result  # return depth

        mm = read_int(16)
        if mm > 2**8:
            mm = -(2**16-mm)  # 负数会出现越界

        interval = read_int(16)

        mode = read_int(2)
        if mode == 1:
            self.mode = 'V'
        elif mode == 2:
            self.mode = 'H'
        elif mode == 0:
            self.mode = None

        H = read_int(16)
        W = read_int(16)
        C = read_int(2)
        # print('decompress:mm,interval,Header',mm,interval,H,W,C)
        self.min = mm
        self.interval = interval
        return H, W, C

    def compress(self, code, bitout):
        enc = HuffmanEncoder(bitout)
        enc.codetree = code
        h, w, c = self.img.shape
        # print("compress img shape",h,w,c)
        size = 0
        # print('hwc',h,w,c)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    # print(self.img[i,j,k])
                    size += enc.write(self.img[i, j, k])
        size += enc.write(self.num)  # EOF
        # print(f"sigs size: {size} bytes")
        return size

    # 直接对赋值进行哈夫曼编码
    def huffman_encodeImg(self, outputfile):
        freqs = self.get_frequencies()
        code = freqs.build_code_tree()
        canoncode = CanonicalCode(
            tree=code, symbollimit=freqs.get_symbol_limit())
        code = canoncode.to_code_tree()

        size = 0
        with contextlib.closing(BitOutputStream(open(outputfile, "wb"))) as bitout:
            size += self.write_code_len_table(bitout, canoncode)
            size += self.writeHeader(bitout)
            size += self.compress(code, bitout)
        h, w, c = self.img.shape
        raw_size = h*w*c
        print(f"raw size: {raw_size} bytes")
        print(f"compress size: {size} bytes")
        print(f"compress ratio: {raw_size/size:.2f}")

        return size

    def main_compress(self, array):
        # 为了适配hash索引，进行-1和+1操作
        st = time.time()
        if self.filename:
            outputfile = result_dir + self.filename+'.bin'
        else:
            outputfile = 'output.bin'
            self.initinize(img_src=array)
        size = self.huffman_encodeImg(outputfile)
        compress_time = time.time()-st
        # print(f'compress time: {compress_time:.4f}s')
        return size, compress_time

    def decompress(self, code, bitin, out):
        H, W, C = self.readHeader(bitin)
        dec = HuffmanDecoder(bitin)
        dec.codetree = code
        img = np.ones([H, W, C])
        for i in range(H):
            for j in range(W):
                for k in range(C):
                    img[i, j, k] = dec.read()
                    if img[i, j, k] != self.img[i, j, k]:
                        print('YUV Image Lossy Compression Failed! At:', i, j, k)
                    # out.write(bytes((symbol,)))  # 写成二进制，还是img.save
        print("YUV Image Lossy Compression Success!")

        img = img + self.min
        self.img = self.img + self.min

        # 是否进行差分编码
        if self.mode == 'V':
            img = self.reverse_dV(img)
            self.img = self.reverse_dV(self.img)
        elif self.mode == 'H':
            img = self.reverse_dH(img)
            self.img = self.reverse_dH(self.img)

        if self.filename:
            dhrgb = Image.fromarray(
                np.array(img, dtype='uint8'), 'YCbCr').convert('RGB')
            rgb = np.array(Image.open(self.fullname).convert('RGB'))
            rgb_mse = ((dhrgb-rgb)**2).sum()/(H*W*C)
            print("RGB MSE:%.4f" % (rgb_mse))
            dhrgb.save(out)
        return img

    def main_decompress(self):
        st = time.time()
        if self.filename:
            outputfile = result_dir+self.filename+'.bmp'
            infile = result_dir+self.filename+'.bin'
        else:
            infile = 'output.bin'
            outputfile = 'output.bmp'

        with open(infile, "rb") as inp, open(outputfile, 'wb') as out:
            bitin = BitInputStream(inp)
            canoncode = self.read_code_len_table(bitin)
            code = canoncode.to_code_tree()
            rec = self.decompress(code, bitin, out)
            bitin.close()
        decompress_time = time.time()-st

        return decompress_time, rec
