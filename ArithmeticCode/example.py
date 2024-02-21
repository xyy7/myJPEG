from random import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import os
import time
from arithmeticcoding import *
import contextlib
import sys
from pathlib import Path


class Example:
    def __init__(self,symbols=range(1,6),prob=[0.6, 0.1, 0.1, 0.1, 0.1],sig_len=10000) -> None:
        self.symbols = symbols
        self.prob = prob # 传入频数还是频率，需要考虑
        self.sig_len = sig_len
        self.size = 0

        # 先生成好固定数字，然后再打乱
        sigs = []
        for i,p in enumerate(prob):
            sigs += [symbols[i]] * int(p*sig_len)
        random.shuffle(sigs)
        self.sigs = sigs
        self.max_depth = int(np.ceil(np.log2(len(prob)+1)))

        if min(prob) < 1:
            for i in range(len(prob)):
                self.prob[i]=int(prob[i]*sig_len)
                
    
    def get_frequencies(self,hist):
        # print(hist)
        return SimpleFrequencyTable(hist+[1])
        # return FrequencyTable(hist)
    
    def write_frequencies(self,bitout, freqs):
        # Writes an unsigned integer of the given bit width to the given stream.
        def write_int(bitout, numbits, value):
            size = 0
            for i in reversed(range(numbits)):
                size+=bitout.write((value >> i) & 1)  # Big endian
            return size 

        size = 0
        interval = len(self.prob)    
        for i in range(interval):
            size+=write_int(bitout, 32, freqs.get(i))
        print(f"freqs size: {size} bytes")
        return size

    def read_frequencies(self,bitin):
        def read_int(n):
            result = 0
            for _ in range(n):
                result = (result << 1) | bitin.read_no_eof()  # Big endian
            return result

        interval = len(self.prob)   
        freqs = [read_int(32) for _ in range(interval)]
        freqs.append(1)  # EOF symbol
        return SimpleFrequencyTable(freqs)
    
    def compressEx(self,freq, sigs, bitout):
        enc = ArithmeticEncoder(32,bitout)
        size = 0
        for sig in sigs:
            size+=enc.write(freq,sig)
        size+=enc.write(freq,len(self.prob))  # EOF 
        print(f"sigs size: {size} bytes")
        enc.finish()
        return size
    
    def arith_encodeEx(self,sigs, prob, outputfile):
        freqs = self.get_frequencies(prob)
    
        size = 0
        with contextlib.closing(BitOutputStream(open(outputfile, "wb"))) as bitout:
            size+=self.write_frequencies(bitout, freqs)  # [1] 不需要写
            size+=self.compressEx(freqs, sigs, bitout)
        print('='*20)
        raw_size = len(sigs)*(np.ceil(np.log2(max(sigs))+1))
        print('AC Encoding Finshed')
        print(f"raw size: {raw_size} bytes")
        print(f"compress size: {size} bytes")
        self.size = size
        print(f"compress ratio: {raw_size/size:.2f}")
        
        return size

    def main_compress(self,ofile='streamEx.bin'):
        # 为了适配hash索引，进行-1和+1操作
        st =time.time()
        frequncy = []
        if min(self.prob) < 1:
            for i in range(len(self.prob)):
                frequncy.append(int(self.prob[i]*self.sig_len))  # 转换为频数
        else:
            frequncy = self.prob 
        self.arith_encodeEx(np.array(self.sigs)-1,frequncy,ofile)
        print(f'compress time: {time.time()-st:.4f}s')
        print('='*20)
    
    def cal_entropy(self,hist,sigs):
        if max(hist)>0:
            for i in range(len(self.prob)):
                hist[i] = hist[i]/self.sig_len
        hist = np.array(hist)
        h = (-hist * np.log2(hist)).sum()
        print(f'Entropy:{h:.2f}')
        # avg_code_length = self.size * 8 / len(sigs)
        # print(f'Average code length:{avg_code_length:.2f}')

    def decompress(self,freqs, bitin, out):
        dec = ArithmeticDecoder(32, bitin)
        dhsig = []
        while True:
            symbol = dec.read(freqs)  # walk through hftree and return a symbol
            if symbol == len(self.prob):  # EOF symbol
                break
            # out.write(bytes((symbol,)))
            dhsig.append(symbol+1)
        # print('sig == dhsig ?:',self.sigs==dhsig[:])
        if self.sigs == dhsig[:]:
            print('AC Encoding Finshed\nAC Lossy Compression Success!')


    def main_decompress(self,ifile='streamEx.bin',out=None):
        st =time.time()
        with open(ifile, "rb") as inp:
            bitin =BitInputStream(inp)
            freqs = self.read_frequencies(bitin)
            self.decompress(freqs, bitin, out)
        print(f'decompress time: {time.time()-st:.4f}s')
        self.cal_entropy(self.prob,self.sigs)
		

ex = Example()
ex.main_compress()
ex.main_decompress()