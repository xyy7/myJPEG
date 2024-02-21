1. dct_coef_and_transform.py: 展示了dct的系数对应的可视化图像，以及dct变换从二维到一维的改进版本。
2. dct_q_and_Qmatrix.py: 展示了固定的量化，以及使用特定的亮度和色度量化的区别。
3. my_jpeg.py: func使用普通版本，直接将量化完的系数通过熵编码（哈夫曼或者算术编码）成二进制；func使用plus版本，则对图像的直流系数进行dpcm，然后进行熵编码，对交流系数进行行程编码rle后，再进行熵编码。