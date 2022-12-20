import os
import random

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import astra
import pylab
import cv2

path = 'D:/Users/vintage/sparse_view_CT_test/original_data/LDCT_DATA/ground_truth_train'
idx = 1
root = path
h5_list = os.listdir(root)
h5_seq = idx // 128
h5_dir = os.path.join(root, h5_list[h5_seq])
tensor_seq = idx % 128
targets = h5py.File(h5_dir, 'r')['data'][tensor_seq]
vol_geom = astra.create_vol_geom(362, 362)
proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                   np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 1000, False),)
P = np.array(targets)
proj_id = astra.create_projector('linear', proj_geom, vol_geom)  # CPU类型的的投影生成模式可以是
                                                             # line, linear, strip, GPU的话使用cuda即可。
sinogram_id, sinogram = astra.create_sino(P, proj_id)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP')  # 进行网络训练的同时使用GPU重建占用大量资源，所以不使用FBP_CUDA
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['option'] = {'FilterType': 'Hann'}  # 如果使用SIRT等迭代算法，此行舍去
cfg['ProjectorId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
#astra.algorithm.run(alg_id, 20)          # 如果使用SIRT等迭代算法，此行指定迭代次数
imgs = astra.data2d.get(rec_id)
sino1 = astra.data2d.get(sinogram_id)
# rows1, cols1 = sino1.shape
# M = cv2.getRotationMatrix2D(((cols1-1)/2.0, (rows1-1)/2.0), 90, 1)
# sino1 = cv2.warpAffine(sino1, M, (cols1, rows1))

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)

vol_geom = astra.create_vol_geom(362, 362)
proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                   np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 60, False),)
P = np.array(targets)
proj_id = astra.create_projector('linear', proj_geom, vol_geom)  # CPU类型的的投影生成模式可以是
                                                             # line, linear, strip, GPU的话使用cuda即可。
sinogram_id, sinogram = astra.create_sino(P, proj_id)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP')  # 进行网络训练的同时使用GPU重建占用大量资源，所以不使用FBP_CUDA
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
cfg['option'] = {'FilterType': 'Hann'}  # 如果使用SIRT等迭代算法，此行舍去
cfg['ProjectorId'] = proj_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
#astra.algorithm.run(alg_id, 20)          # 如果使用SIRT等迭代算法，此行指定迭代次数
imgs = astra.data2d.get(rec_id)
sino2 = astra.data2d.get(sinogram_id)
# rows2, cols2 = sino2.shape
# M = cv2.getRotationMatrix2D(((cols2-1)/2.0, (rows2-1)/2.0), 90, 1)
# sino2 = cv2.warpAffine(sino2, M, (cols2, rows2))

astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
astra.data2d.delete(sinogram_id)
astra.projector.delete(proj_id)


pylab.gray()
pylab.figure(1)
pylab.axis('off')
pylab.xticks([])
pylab.yticks([])
pylab.imshow(sino1)
pylab.colorbar()
pylab.figure(2)
pylab.axis('off')
pylab.xticks([])
pylab.yticks([])
pylab.imshow(sino2)
pylab.colorbar()
pylab.show()