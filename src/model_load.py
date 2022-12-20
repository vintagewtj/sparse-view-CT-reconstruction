import time

import pylab
import torch

from dataset_generator import *
from model import *
device = torch.device("cuda:0")

epoch = 19                       # 指定使用训练第几个epoch后所保存的模型进行测试
i = 20                         # 指定使用训练集中第几个数据进行测试
# test_data = Datasets('/mnt/PyCharm_Project_1/original_data/LDCT_DATA/ground_truth_test')
test_data = Datasets('D:/Users/vintage/sparse_view_CT_test/original_data/LDCT_DATA/ground_truth_test')
# root = '2022.4.19'
if not os.path.exists('../outputs'):
    os.makedirs('../outputs')
save_path = "../outputs/model{0}_outputs{1}.bmp".format(epoch, i)

test_img = test_data[i][0][0]

# vol_geom = astra.create_vol_geom(362, 362)
# proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
#                                     np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 60, False))
# P = np.array(test_img)
# proj_id = astra.create_projector('linear', proj_geom, vol_geom)
# sinogram_id, sinogram = astra.create_sino(P, proj_id)
# rec_id = astra.data2d.create('-vol', vol_geom)
# cfg = astra.astra_dict('FBP')  # FBP_CUDA
# cfg['ReconstructionDataId'] = rec_id
# cfg['ProjectionDataId'] = sinogram_id
# cfg['option'] = {'FilterType': 'Hann'}
# cfg['ProjectorId'] = proj_id
#
# alg_id = astra.algorithm.create(cfg)
# astra.algorithm.run(alg_id)
# # astra.algorithm.run(alg_id, 20)
# test_img = astra.data2d.get(rec_id)
# transforms = torchvision.transforms.ToTensor()
# test_img = transforms(test_img)

start_time = time.time()
input_img = test_img.view(1, 1, 362, 362)
model = torch.load("../checkpoints/model_{0}.pth".format(epoch), )
model.to(device)
with torch.no_grad():
    input_img = input_img.to(device)
    output = model(input_img)

# print(output)
output = output.cpu()
out_img = np.array(output)
out_img = np.reshape(out_img, (362, 362))
end_time = time.time()
print("加载时间：{}".format(end_time - start_time))
# slope = (1-(0))/(np.max(out_img)-np.min(out_img))
# out_img = (0) + slope*(out_img-np.min(out_img))
pylab.gray()
pylab.figure(1)
pylab.imshow(out_img)
pylab.colorbar()
pylab.show()
pylab.imsave(save_path, out_img)