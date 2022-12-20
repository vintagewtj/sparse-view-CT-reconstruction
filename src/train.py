import torch
import torchvision
from torch import nn, cuda
import torch.multiprocessing as mp
# from torch.cuda.amp import autocast as autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import *       # import model 的话创建网络就要写成 model = model.Net()
from dataset_generator import *
# import adabound

# 准备数据集
train_data = Datasets('/root/autodl-tmp/original_data/LDCT_DATA/ground_truth_train')
test_data = Datasets('/root/autodl-tmp/original_data/LDCT_DATA/ground_truth_test')
# train_data = Datasets('/mnt/PyCharm_Project_1/original_data/LDCT_DATA/ground_truth_train')
# test_data = Datasets('/mnt/PyCharm_Project_1/original_data/LDCT_DATA/ground_truth_test')

# 查看数据集大小 length函数
train_data_size = len(train_data)
test_data_size = len(test_data)
# print("训练数据集的长度为:{}".format(train_data_size))
# print("测试数据集的长度为:{}".format(test_data_size))

# 利用dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True) #num_workers=4
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # 创建网络模型
    model = Net()
    device = torch.device("cuda:0")
    model.to(device)
    # model.apply(weight_init)

    # 损失函数
    loss_fn = nn.MSELoss(reduction='mean')
    loss_fn.to(device)

    # 优化器
    learning_rate = 1e-5  # learning_rate = 0.01   1e-2 = (10)^(-2) = 0.01
    regulation_penalty = 1e-7
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regulation_penalty)  #adabound.AdaBound, weight_decay=regulation_penalty
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)


    # 设置训练网络的一些参数
    total_train_step = 0  # 记录训练的次数
    total_test_step = 0  # 记录测试的次数
    epoch = 20  # 训练的轮数

    # 添加tensorboard
    if not os.path.exists('../logs_train/train'):
        os.makedirs('../logs_train/train')
    if not os.path.exists('../logs_train/test'):
        os.makedirs('../logs_train/test')
    writer = SummaryWriter("../logs_train")
    train_writer = SummaryWriter("../logs_train/train")
    test_writer = SummaryWriter("../logs_train/test")

    # 在训练最开始之前实例化一个GradScaler对象--用于混合精度运算
    # scaler = GradScaler()

    # 训练步骤开始
    model.train()
    start_time = time.time()
    for i in range(epoch):
        print("----------第 {} 轮训练开始----------".format(i + 1))
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            # optimizer.zero_grad()
            # 前向过程(model + loss)开启 autocast
            # with autocast():
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            # 1、Scales loss.  先将梯度放大 防止梯度消失
            # scaler.scale(loss).backward()

            # 2、scaler.step()   再把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            # scaler.step(optimizer)

            # 3、准备着，看是否要增大scaler
            # scaler.update()
            # 正常更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print("训练时间：{}".format(end_time - start_time))
                print("训练次数: {0}，Loss: {1}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                train_writer.add_scalar("loss", loss.item(), total_train_step)

        print("第%d个epoch的学习率：%f" % (i, optimizer.param_groups[0]['lr']))
        # scheduler.step()

        # 测试步骤开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        iter_num = 0
        with torch.no_grad():  # 查看模型训练时不生成计算图
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                iter_num += 1
            mean_test_loss = total_test_loss / iter_num

        total_test_step += 1
        print("整体测试集上的Loss: {}".format(mean_test_loss))
        writer.add_scalar("test_loss", mean_test_loss, total_test_step)
        test_writer.add_scalar("loss", mean_test_loss, total_train_step)


        if not os.path.exists('../checkpoints'):
            os.makedirs('../checkpoints')
        torch.save(model, "../checkpoints/model_{}.pth".format(i))
        # 官方推荐模型保存方式： torch.save(model.state_dict(), "./pth/model_{}.pth".format(i))
        print("模型已保存")

    writer.close()
