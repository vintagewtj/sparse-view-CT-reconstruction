import os

import cv2
import numpy as np
# from skimage import metrics
from scipy import signal
from dataset_generator import *
from model import *
from evaluation import *
from evaluation import _ssim_one_channel
device = torch.device("cuda:0")

test_data = Datasets('D:/Users/vintage/sparse_view_CT_test/original_data/LDCT_DATA/ground_truth_test')

#epoch_list = os.listdir('../run_records/{}/checkpoints'.format(root))
for epoch in range(0, 20):        # 指定使用训练第几个epoch后所保存的模型进行测试
    model = torch.load("../checkpoints/model_{0}.pth".format(epoch))  # /run_records/2022.3.28_2Unet_b_s16_epoch20_60angles
    model.to(device)
    # epoch = 19
    mse_sum = 0
    psnr_sum = 0
    ssim_sum = 0
    mse_list = []
    psnr_list = []
    ssim_list = []
    test_list = np.random
    test_num = 10

    for i in range(test_num):                   # 指定使用训练集中第几个数据进行测试
        # test_img = test_data[i][1][0]
        #
        # vol_geom = astra.create_vol_geom(362, 362)
        # proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
        #                                    np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 1000, False))
        # P = np.array(test_img)
        #
        # proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
        # sinogram_id, sinogram = astra.create_sino(P, proj_id)
        # rec_id = astra.data2d.create('-vol', vol_geom)
        # cfg = astra.astra_dict('FBP_CUDA')  # FBP_CUDA
        # cfg['ReconstructionDataId'] = rec_id
        # cfg['ProjectionDataId'] = sinogram_id
        # cfg['option'] = {'FilterType': 'Hann'}
        # cfg['ProjectorId'] = proj_id
        #
        # alg_id = astra.algorithm.create(cfg)
        # astra.algorithm.run(alg_id)
        #
        # gt_img = astra.data2d.get(rec_id)
        # astra.algorithm.delete(alg_id)
        # astra.data2d.delete(rec_id)
        # astra.data2d.delete(sinogram_id)
        # astra.projector.delete(proj_id)

        gt_img = test_data[i][1][0]
        gt_img = np.array(gt_img)
        gt_img = np.reshape(gt_img, (362, 362))
        input_img = test_data[i][0][0]
        input_img_tensor = input_img
        # input_img = np.array(input_img)
        # input_img = np.reshape(input_img, (362, 362))
        input_img_tensor = input_img_tensor.view(1, 1, 362, 362)

        with torch.no_grad():
            input_img_tensor = input_img_tensor.to(device)
            output = model(input_img_tensor)

        output = output.cpu()
        out_img = np.array(output)
        out_img = np.reshape(out_img, (362, 362))

        # slope1 = (255-(0))/(np.max(P)-np.min(P))
        # P = (0) + slope1 * (P-np.min(P))
        # slope2 = (255 - (0)) / (np.max(out_img) - np.min(out_img))
        # out_img = (0) + slope2 * (out_img - np.min(out_img))

        ran = np.max(gt_img) - np.min(gt_img)
        mse = compute_mse(out_img, gt_img)
        psnr = compute_psnr(out_img, gt_img, ran)
        ssim = _ssim_one_channel(out_img, gt_img, 11, ran)
        # mse = compute_mse(input_img, gt_img)
        # psnr = compute_psnr(input_img, gt_img, ran)
        # ssim = _ssim_one_channel(input_img, gt_img, 11, ran)
        mse_sum += mse
        psnr_sum += psnr
        ssim_sum += ssim
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        # astra.algorithm.delete(alg_id)
        # astra.data2d.delete(rec_id)
        # astra.data2d.delete(sinogram_id)
        # astra.projector.delete(proj_id)

    mse_mean = mse_sum / test_num
    mse_var = np.var(mse_list)
    psnr_mean = psnr_sum / test_num
    psnr_var = np.var(psnr_list)
    ssim_mean = ssim_sum / test_num
    ssim_var = np.var(ssim_list)
    print("epoch{0}_mse:{1} +- {2}".format(epoch, mse_mean, mse_var))
    print("epoch{0}_psnr:{1} +- {2}".format(epoch, psnr_mean, psnr_var))
    print("epoch{0}_ssim:{1} +- {2}\n".format(epoch, ssim_mean, ssim_var))

    # pylab.gray()
    # pylab.figure(1)
    # pylab.imshow(P)
    # pylab.colorbar()
    # pylab.figure(2)
    # pylab.imshow(out_img)
    # pylab.colorbar()
    # pylab.show()