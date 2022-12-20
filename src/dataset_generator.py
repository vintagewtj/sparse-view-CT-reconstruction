import os
import random

import h5py
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import astra
import pylab
import cv2

class Datasets(Dataset):
    def __init__(self, path):
        super(Datasets, self).__init__()
        self.root = path
        self.h5_list = os.listdir(self.root)
        self.dataset = None

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, idx):
        if self.dataset is None:
            self.h5_seq = idx // 128
            self.h5_dir = os.path.join(self.root, self.h5_list[self.h5_seq])
            self.tensor_seq = idx % 128
            targets = h5py.File(self.h5_dir, 'r')['data'][self.tensor_seq]
            vol_geom = astra.create_vol_geom(362, 362)
            proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                               np.linspace(np.pi * -110 / 180, np.pi * 110 / 180, 90, False,),)
            P = np.array(targets)
            print(P)
            proj_id = astra.create_projector('cuda', proj_geom, vol_geom)  # CPU类型的的投影生成模式可以是
                                                                         # line, linear, strip, GPU的话使用cuda即可。
            sinogram_id, sinogram = astra.create_sino(P, proj_id)
            rec_id = astra.data2d.create('-vol', vol_geom)
            cfg = astra.astra_dict('FBP_CUDA')  # 进行网络训练的同时使用GPU重建占用大量资源，所以不使用FBP_CUDA
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sinogram_id
            cfg['option'] = {'FilterType': 'Hann'}  # 如果使用SIRT等迭代算法，此行舍去
            cfg['ProjectorId'] = proj_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            #astra.algorithm.run(alg_id, 20)          # 如果使用SIRT等迭代算法，此行指定迭代次数
            imgs = astra.data2d.get(rec_id)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sinogram_id)
            astra.projector.delete(proj_id)

            vol_geom = astra.create_vol_geom(362, 362)
            proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                               np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 2000, False), )
            P = np.array(targets)
            proj_id = astra.create_projector('cuda', proj_geom, vol_geom)  # CPU类型的的投影生成模式可以是
            # line, linear, strip, GPU的话使用cuda即可。
            sinogram_id, sinogram = astra.create_sino(P, proj_id)
            rec_id = astra.data2d.create('-vol', vol_geom)
            cfg = astra.astra_dict('FBP_CUDA')  # 进行网络训练的同时使用GPU重建占用大量资源，所以不使用FBP_CUDA
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sinogram_id
            cfg['option'] = {'FilterType': 'Hann'}  # 如果使用SIRT等迭代算法，此行舍去
            cfg['ProjectorId'] = proj_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            # astra.algorithm.run(alg_id, 20)          # 如果使用SIRT等迭代算法，此行指定迭代次数
            targets = astra.data2d.get(rec_id)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sinogram_id)
            astra.projector.delete(proj_id)

            # 随机进行数据增强，为2时不做处理
            # flipCode = random.choice([-1, 0, 1, 2])
            # if flipCode != 2:
            #     imgs = self.augment(imgs, flipCode)
            #     targets = self.augment(targets, flipCode)
            # else:
            #     imgs = imgs
            #     targets = targets

            transform = transforms.ToTensor()
            imgs = transform(imgs)
            imgs = imgs.view(1, 362, 362)
            targets = transform(targets)
            targets = targets.view(1, 362, 362)


        return imgs, targets


    def __len__(self):
        return len(self.h5_list) * 128

if __name__ == '__main__':
    # train_data = Datasets('autodl-tmp/original_data/LDCT_DATA/ground_truth_train')
    # test_data = Datasets('autodl-tmp/original_data/LDCT_DATA/ground_truth_test')

    train_data = Datasets('D:/Users/vintage/sparse_view_CT_test/original_data/LDCT_DATA/ground_truth_train')
    test_data = Datasets('D:/Users/vintage/sparse_view_CT_test/original_data/LDCT_DATA/ground_truth_test')

    # train_data = Datasets('/mnt/PyCharm_Project_1/original_data/LDCT_DATA/ground_truth_train')
    # test_data = Datasets('/mnt/PyCharm_Project_1/original_data/LDCT_DATA/ground_truth_test')
    i = 15
    a, b = test_data[i]
    # b = test_data[4][1]
    a = a.view(362, 362)
    b = b.view(362, 362)
    pylab.gray()
    pylab.figure(1)
    pylab.imshow(a)
    pylab.colorbar()
    pylab.figure(2)
    pylab.imshow(b)
    pylab.colorbar()
    pylab.show()

