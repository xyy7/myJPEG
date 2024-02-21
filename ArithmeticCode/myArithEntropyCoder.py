import contextlib
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .arithmeticcoding import *

result_dir = 'arith_compress/'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

totol_bits = 32  # 位数太少，可能会导致解码越界；解码不越界了，可能还是匹配不上


class Image2bin:
    def __init__(self, mode=None, is8bit=False, filename=None) -> None:
        self.is8bit = is8bit
        self.max_depth = int(np.ceil(np.log2(256 if is8bit else 512)))
        self.mode = mode
        if filename is not None:
            self.fullname = filename
            self.filename = Path(filename).stem
            self.img = np.array(Image.open(filename).convert('YCbCr'))
            self.initinize(self.img)
        else:
            self.fullname = None
            self.filename = None
            self.img = None

    def initinize(self, img):
        if self.mode == 'V':
            img = self.difference_imgV(img)
        elif self.mode == 'H':
            img = self.difference_imgH(img)
        else:
            pass

        if len(img.shape) == 2:
            img = img[:, :, None]

        self.img = img
        self.hist = list(self.get_hist())
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

    def get_hist(self, img=None, is8bit=None):
        interval = 256 if self.is8bit else 511
        x, y, c = self.img.shape
        hist = np.zeros(interval)
        if self.is8bit:
            for i in range(x):
                for j in range(y):
                    for k in range(c):
                        hist[self.img[i, j, k]] += 1
        else:
            for i in range(x):
                for j in range(y):
                    for k in range(c):
                        hist[self.img[i, j, k]+255] += 1

        return hist

    def get_frequencies(self):
        return SimpleFrequencyTable(self.hist+[1])

    def write_frequencies(self, bitout, freqs):
        # Writes an unsigned integer of the given bit width to the given stream.
        def write_int(bitout, numbits, value):
            size = 0
            for i in reversed(range(numbits)):
                size += bitout.write((value >> i) & 1)  # Big endian
            return size

        size = 0
        interval = 256 if self.is8bit else 511
        for i in range(interval):
            size += write_int(bitout, 32, freqs.get(i))
        print(f"freqs size: {size} bytes")
        return size

    def writeHWC(self, bitout):
        # print('img shape:',self.img.shape)
        # h 2B,w 2B,c 2b
        size = 0
        h, w, c = self.img.shape
        for j in reversed(range(16)):
            size += bitout.write((h >> j) & 1)

        for j in reversed(range(16)):
            size += bitout.write((w >> j) & 1)

        for j in reversed(range(2)):
            size += bitout.write((c >> j) & 1)
        print('H:2B,W:2B,C:2bit')
        return size

    def read_frequencies(self, bitin):
        def read_int(n):
            result = 0
            for _ in range(n):
                result = (result << 1) | bitin.read_no_eof()  # Big endian
            return result

        interval = 256 if self.is8bit else 511
        freqs = [read_int(32) for _ in range(interval)]  # 0-510
        freqs.append(1)  # EOF symbol                    # 511
        freqs = SimpleFrequencyTable(freqs)
        return freqs

    def readHWC(self, bitin):
        def read_int(n):
            result = 0
            for _ in range(n):
                result = (result << 1) | bitin.read_no_eof()  # Big endian
            return result  # return depth

        H = read_int(16)
        W = read_int(16)
        C = read_int(2)
        print('decompress:HWC', H, W, C)
        return H, W, C

    def compress(self, freq, bitout):
        interval = 256 if self.is8bit else 511
        enc = ArithmeticEncoder(totol_bits, bitout)
        h, w, c = self.img.shape
        print("compress img shape", h, w, c)
        # for i in range(freq.get_symbol_limit()):
        #     print(i-255,freq.get(i))

        size = 0
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    offset = 0 if self.is8bit else 255
                    size += enc.write(freq, self.img[i, j, k]+offset)
        size += enc.write(freq, interval)  # EOF
        enc.finish()
        print(f"sigs size: {size} bytes")
        return size

    def arith_encodeImg(self, outputfile):
        freqs = self.get_frequencies()
        size = 0
        with contextlib.closing(BitOutputStream(open(outputfile, "wb"))) as bitout:
            size += self.write_frequencies(bitout, freqs)
            size += self.writeHWC(bitout)
            size += self.compress(freqs, bitout)
        print('='*20)
        h, w, c = self.img.shape
        raw_size = h*w*c
        print(f"raw size: {raw_size} bytes")
        print(f"compress size: {size} bytes")
        print(f"compress ratio: {raw_size/size:.2f}")

        return size

    def main_compress(self, array=None):
        # 为了适配hash索引，进行-1和+1操作
        st = time.time()
        if array:
            self.initinize(array)
        outputfile = self.filename+'.bin'
        size = self.arith_encodeImg(outputfile)

        compress_time = time.time()-st
        print(f'compress time: {compress_time:.4f}s')
        print('='*20)
        return size, compress_time

    def decompress(self, freqs, bitin, out):
        H, W, C = self.readHWC(bitin)
        dec = ArithmeticDecoder(totol_bits, bitin)
        img = np.ones([H, W, C])
        offset = 0 if self.is8bit else 255

        for i in range(H):
            for j in range(W):
                for k in range(C):
                    img[i, j, k] = dec.read(freqs)-offset
                    if img[i, j, k] != self.img[i, j, k]:
                        print('YUV Image Lossy Compression Failed! At:', i, j, k,
                              f"{self.img[i,j,k]},{img[i,j,k]-self.img[i,j,k]}")  # 第一个解码有问题，后面的都会有问题
                        return None
                    # out.write(bytes((symbol,)))  # 写成二进制，还是img.save
        print("YUV Image Lossy Compression Success!")
        if self.mode == 'V':
            img = self.reverse_dV(img)
            self.img = self.reverse_dV(self.img)
        elif self.mode == 'H':
            img = self.reverse_dH(img)
            self.img = self.reverse_dH(self.img)

        rec = Image.fromarray(
            np.array(img, dtype='uint8'), 'YCbCr').convert('RGB')
        rgb = np.array(Image.open(self.fullname).convert('RGB'))
        rec.save(out)
        return rec

    def main_decompress(self):
        st = time.time()
        outputfile = self.filename+'.bmp'
        with open(self.filename+'.bin', "rb") as inp, open(outputfile, 'wb') as out:
            bitin = BitInputStream(inp)
            freqs = self.read_frequencies(bitin)
            rec = self.decompress(freqs, bitin, out)
        decompress_time = time.time()-st
        print(f'decompress time: {decompress_time:.4f}s')
        return decompress_time, rec
