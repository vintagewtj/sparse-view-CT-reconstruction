import time

import astra
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pylab
from model import *
from evaluation import *
from evaluation import _ssim_one_channel


device = torch.device("cuda:0")

img = Image.open(r'D:\Users\vintage\sparse_view_CT_test\sample\up.bmp')
resize = transforms.Resize(362)
img = resize(img)
slope = (1-(0))/(np.max(img)-np.min(img))
img= (0) + slope * (img-np.min(img))
transforms = transforms.ToTensor()

for epoch in range(0, 20):
    # epoch = 19
    # root = '2022.3.30_Unet_bs16_epoch20_60angles_Adam'

    vol_geom = astra.create_vol_geom(362, 362)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                        np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 2000, False))
    P = np.array(img)
    # slope = (1-(0))/(np.max(P)-np.min(P))
    # P = (0) + slope*(P-np.min(P))

    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(P, proj_id)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')  # FBP_CUDA
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {'FilterType': 'Hann'}
    cfg['ProjectorId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    # astra.algorithm.run(alg_id, 20)
    gt_img = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    start_time = time.time()
    vol_geom = astra.create_vol_geom(362, 362)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                        np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 60, False))
    P = np.array(img)
    # slope = (1-(0))/(np.max(P)-np.min(P))
    # P = (0) + slope*(P-np.min(P))

    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    sinogram_id, sinogram = astra.create_sino(P, proj_id)
    rec_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')  # FBP_CUDA
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {'FilterType': 'Hann'}
    cfg['ProjectorId'] = proj_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    # astra.algorithm.run(alg_id, 20)
    test_img = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)
    end_time = time.time()
    start_time2 = time.time()
    input_img = transforms(test_img)

    input_img = input_img.view(1, 1, 362, 362)
    model = torch.load("../checkpoints/model_{0}.pth".format(epoch), )
    model.to(device)
    with torch.no_grad():
        input_img = input_img.to(device)
        output = model(input_img)
    output = output.cpu()
    out_img = np.array(output)
    out_img = np.reshape(out_img, (362, 362))
    end_time2 = time.time()
    ran = np.max(gt_img) - np.min(gt_img)
    mse1 = compute_mse(gt_img, test_img)
    psnr1 = compute_psnr(gt_img, test_img, ran)
    ssim1 = _ssim_one_channel(gt_img, test_img, 11, ran)

    mse2 = compute_mse(gt_img, out_img)
    psnr2 = compute_psnr(gt_img, out_img, ran)
    ssim2 = _ssim_one_channel(gt_img, out_img, 11, ran)

    print("FBP时间：{}".format(end_time - start_time))
    print("post-processing时间：{}".format(end_time2 - start_time2))
    print('FBP:\nmse:{0}\npsnr:{1}\nssim:{2}\n'.format(mse1, psnr1, ssim1))
    print('epoch{0}: FBP+Unet:\nmse:{1}\npsnr:{2}\nssim:{3}\n'.format(epoch, mse2, psnr2, ssim2))

# pylab.gray()
# pylab.figure(1)
# pylab.imshow(gt_img)
# pylab.axis('off')
# pylab.xticks([])
# pylab.yticks([])
# # pylab.colorbar()
# pylab.figure(2)
# pylab.imshow(test_img)
# # pylab.colorbar()
# pylab.figure(3)
# pylab.imshow(out_img)
# # pylab.colorbar()
# pylab.show()