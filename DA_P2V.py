import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
# model_zoopytypython,
import torch.utils.data as data
import torch.optim as optim
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output
from utils import *
from discriminator_enhance import FCDiscriminator
from discriminator_feature import FCDiscriminator_feature
from deeplab_enhanceCross import DeeplabMulti
import time
import torch.autograd as autograd
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener


# Parameters
RESTORE_FROM = './ISPRS_dataset/resnet101-5d3b4d8f.pth'
WINDOW_SIZE = (256, 256) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "./ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE =6 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

MAIN_FOLDER_P = FOLDER + 'potsdam/'
DATA_FOLDER_P = MAIN_FOLDER_P + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
LABEL_FOLDER_P = MAIN_FOLDER_P + '5_Labels_for_participants/top_potsdam_{}_label.tif'
ERODED_FOLDER_P = MAIN_FOLDER_P + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

MAIN_FOLDER_V = FOLDER + 'vaihingen/'
DATA_FOLDER_V = MAIN_FOLDER_V + 'top/top_mosaic_09cm_area{}.tif'
LABEL_FOLDER_V = MAIN_FOLDER_V + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER_V = MAIN_FOLDER_V + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

# net = ResUnetPlusPlus(3).cuda()
net = DeeplabMulti(num_classes=N_CLASSES)
#初始化一个变量params用于统计模型参数的总数。
#遍历net模型的所有参数，使用param.nelement()方法计算每个参数的元素数量（即参数的大小），
# 并累加到params变量中。
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('Params: ', params)
model_D1 = FCDiscriminator(num_classes=N_CLASSES)
model_D2 = FCDiscriminator(num_classes=N_CLASSES)
model_Df = FCDiscriminator_feature()

net = net.cuda()
model_D1 = model_D1.cuda()
model_D2 = model_D2.cuda()
model_Df = model_Df.cuda()

train_ids_P = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
             '4_12', '6_8', '6_12', '6_7', '4_11']
test_ids_P = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
train_ids_V = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids_V = ['5', '21', '15', '30']

print("Potsdam for training : ", train_ids_P)
print("Potsdam for testing : ", test_ids_P)
print("Vaihingen for training : ", train_ids_V)
print("Vaihingen for testing : ", test_ids_V)
DATASET_P = 'Potsdam'
DATASET_V = 'Vaihingen'
train_set = ISPRS_dataset(train_ids_P, train_ids_V, DATASET_P, DATASET_V, DATA_FOLDER_P, DATA_FOLDER_V,
                          LABEL_FOLDER_P, LABEL_FOLDER_V, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

#定义了一些超参数，如各种损失函数的权重
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
LAMBDA_ADV_DF = 0.005
LAMBDA_MMD = 2.0
if RESTORE_FROM[:4] == 'http':
    saved_state_dict = model_zoo.load_url(RESTORE_FROM)
else:
    saved_state_dict = torch.load(RESTORE_FROM)

new_params = net.state_dict().copy()

#加载一个预训练模型的参数到一个新的模型中
#这里应该是创建了一个新的空的权重表，然后把权重文件处理后加入了进去
for i in saved_state_dict:
    # Scale.layer5.conv2d_list.3.weight
    i_parts = i.split('.')
    # print i_parts
    if not N_CLASSES == 6 or not i_parts[1] == 'layer5':
        new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        # print i_parts
#
net.load_state_dict(new_params,False)

#初始化了四个优化器：optimizer用于优化net模型的参数，
# optimizer_D1、optimizer_D2和optimizer_Df分别用于优化三个判别器模型的参数。
optimizer = optim.SGD(net.optim_parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
#调用了zero_grad()方法，用于清空优化器中的梯度信息，这是在每次迭代开始前通常需要进行的操作。
optimizer.zero_grad()

optimizer_D1 = optim.Adam(model_D1.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
optimizer_D2 = optim.Adam(model_D2.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
optimizer_Df = optim.Adam(model_Df.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
optimizer_D1.zero_grad()
optimizer_D2.zero_grad()
optimizer_Df.zero_grad()

#初始化了一个二元交叉熵损失函数bce_loss，通常用于二分类问题。
bce_loss = torch.nn.BCEWithLogitsLoss()
interp = nn.Upsample(size=(256, 256), mode='bilinear')

source_label = 0
target_label = 1

def test(test_ids_P, test_ids_V, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    #准备测试集数据，包括测试图像、标签和经过侵蚀处理的标签。
    test_imagess = (1 / 255 * np.asarray(io.imread(DATA_FOLDER_P.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids_P)
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER_V.format(id)), dtype='float32') for id in test_ids_V)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER_V.format(id)), dtype='uint8') for id in test_ids_V)
    eroded_labels = (convert_from_color(io.imread(LABEL_FOLDER_V.format(id))) for id in test_ids_V)

    # eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    #初始化两个空列表all_preds和all_gts，用于存储所有的预测结果和真实标签。
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    #将网络切换到推理模式（net.eval()），并禁用梯度计算（torch.no_grad()），以节省内存并提高推理速度。
    net.eval()
    feature_map = False
    with torch.no_grad():
        #遍历测试图像，对于每张图像，初始化一个全零的预测数组pred。
        #使用滑动窗口方法将图像分割成小块，并对每个小块进行推理。
        #将推理结果（经过上采样和Softmax处理）填充到pred数组中对应的位置。
        #对pred数组进行最大值索引操作，得到最终的预测类别。
        #将预测结果和真实标签添加到all_preds和all_gts列表中。
            """
            1.zip(test_imagess, test_images, test_labels, eroded_labels)：这个函数将四个生成器
            test_imagess、test_images、test_labels和eroded_labels打包成一个迭代器，每次迭代会同时从这四个生成器中各取出一个元素，形成一个元组。
            2.tqdm(...)：tqdm是一个快速、可扩展的Python进度条库，用于在长循环中显示进度。在这里，它将zip函数返回的迭代器包装起来，
            创建一个新的迭代器，每次迭代时会更新并显示进度条。
            3.for imgs, imgt, gt, gt_e in ...：这是实际的循环语句，每次迭代会从tqdm包装的迭代器中取出一个元组，元组包含四个元素，
            分别赋值给变量imgs、imgt、gt和gt_e。这些变量分别代表当前迭代的测试图像、目标图像、标签和侵蚀标签。
            """
            for imgs, imgt, gt, gt_e in tqdm(zip(test_imagess, test_images, test_labels, eroded_labels), total=len(test_ids_V), leave=False):
                pred = np.zeros(imgt.shape[:2] + (N_CLASSES,))

                #计算滑动窗口的数量，即图像可以被分割成多少个大小为window_size的小块，然后除以批次大小batch_size得到总的批次数量。
                total = count_sliding_window(imgt, step=stride, window_size=window_size) // batch_size
                #使用sliding_window函数生成图像的滑动窗口坐标，grouper函数将这些坐标分组为批次大小batch_size的批次。
                for i, coords in enumerate(
                        tqdm(grouper(batch_size, sliding_window(imgt, step=stride, window_size=window_size)), total=total,
                                   leave=False)):

                    #构建张量：对于每个批次的坐标，从源图像imgs和目标图像imgt中提取对应的小块，并将其通道维度移到前面，以符合PyTorch的格式要求。
                    #将提取的小块转换为NumPy数组，然后转换为PyTorch的Variable，并移动到GPU上（.cuda()）。
                    # Build the tensor
                    coordss = ((100, 100, 256, 256),)
                    # coordss = ((100, 100, 256, 256))
                    image_patchess = [np.copy(imgs[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coordss]
                    image_patchess = np.asarray(image_patchess)
                    image_patchess = Variable(torch.from_numpy(image_patchess).cuda())
                    image_patches = [np.copy(imgt[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = Variable(torch.from_numpy(image_patches).cuda())

                # Do the inference
                _, _, _, pred_target2, _, atty, _, _ = net(image_patchess, image_patches)
                pred2 = F.softmax(pred_target2, dim=1)
                outs = interp(pred2)
                if feature_map:
                    # x_comp = 80
                    # y_comp = 20
                    # pred = outs[:, 1, x_comp, y_comp]

                    x_comp = 50
                    y_comp = 100
                    pred = outs[:, 4, x_comp, y_comp]
                    feature = outs
                    feature_grad = autograd.grad(pred, feature, allow_unused=True, retain_graph=True)[0]
                    
                    grads = feature_grad  # 获取梯度
                    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
                    # 此处batch size默认为1，所以去掉了第0维（batch size维）
                    pooled_grads = pooled_grads[0]
                    feature = feature[0]
                    # print("pooled_grads:", pooled_grads.shape)
                    # print("feature:", feature.shape)
                    # feature.shape[0]是指定层feature的通道数
                    for i in range(feature.shape[0]):
                        feature[i, ...] *= pooled_grads[i, ...]

                    heatmap = feature.detach().cpu().numpy()
                    heatmap = np.mean(heatmap, axis=0)
                    heatmap1 = np.maximum(heatmap, 0)
                    heatmap1 /= np.max(heatmap1)
                    heatmap1 = cv2.resize(heatmap1, (256, 256))
                    # heatmap[heatmap < 0.7] = 0
                    heatmap1 = np.uint8(255 * heatmap1)
                    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
                    heatmap1 = heatmap1[:, :, (2, 1, 0)]

                    fig = plt.figure()
                    fig.add_subplot(1, 2, 1)
                    image_patches = np.asarray(255 * torch.squeeze(image_patches).cpu(), dtype='uint8').transpose((1, 2, 0))
                    plt.imshow(image_patches)
                    plt.axis('off')
                    plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))

                    fig.add_subplot(1, 2, 2)
                    plt.imshow(heatmap1)
                    plt.gca().add_patch(plt.Rectangle((x_comp - 2, y_comp - 2), 2, 2, color='red', fill=False, linewidth=1))
                    plt.axis('off')
                    plt.savefig('CSATAGAN_car.pdf', dpi=1200)
                    plt.show()

                outs = outs.data.cpu().numpy()

                # Fill in the results array-
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            clear_output()
            all_preds.append(pred)
            all_gts.append(gt_e)

            clear_output()
    #计算所有预测结果的准确率。
    #将网络切换回训练模式（net.train()）。
    #根据条件返回准确率或者准确率以及所有的预测结果和真实标签。
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    net.train()
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

def train(epochs, weights=WEIGHTS, save_epoch=2):
    weights = weights.cuda()
    MIoU_best = 0.55
    # MIoU_best = 0
    Name_best = ''
    iter = 0
    # 将网络模型和判别器模型设置为训练模式。
    net.train()
    model_D1.train()
    model_D2.train()
    model_Df.train()
    #外层循环遍历每个epoch。
    #内层循环遍历每个batch的数据。
    #在每个batch开始时，记录开始时间，并清空优化器的梯度。
    for epoch in range(1, epochs + 1):
        for batch_idx, (images, labels, images_t, labels_t) in enumerate(train_loader):
            start_time = time.time()
            optimizer.zero_grad()
            #使用 adjust_learning_rate 函数调整主网络模型的学习率。
            adjust_learning_rate(optimizer, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer_Df.zero_grad()
            #使用 adjust_learning_rate_D 函数调整判别器模型的学习率。
            adjust_learning_rate_D(optimizer_D1, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_D2, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_Df, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            # train G
            # don't accumulate grads in D
            # 将判别器模型的参数设置为不需要梯度，以防止在训练生成器时更新判别器的参数。
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            for param in model_Df.parameters():
                param.requires_grad = False

            # train with source and target
            #将源域和目标域的图像数据移到GPU上，并通过网络模型得到预测结果和注意力图。
            images = Variable(images).cuda()
            images_t = Variable(images_t).cuda()
            pred1, pred2, pred_target1, pred_target2, attx, atty, attxp, attyp = net(images, images_t)


        #计算多种损失，包括分割损失、对抗损失和MMD（Maximum Mean Discrepancy）损失。
            #计算源域和目标域注意力图之间的MMD损失
            loss_mmd = discrepancy(attxp, attyp)
            ## coral
            # loss_mmd = coral(attxp, attyp)
            #对预测结果进行插值操作，以匹配标签的尺寸。
            pred1 = interp(pred1)
            pred2 = interp(pred2)
            #这里的损失计算可能涉及到了权重，以处理类别不平衡等问题
            loss_seg1 = loss_calc(pred1, labels, weights)
            loss_seg2 = loss_calc(pred2, labels, weights)
            pred_target1 = interp(pred_target1)
            pred_target2 = interp(pred_target2)
            #将经过softmax处理的预测结果输入到判别器模型中，得到判别器的输出
            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))
            D_f = model_Df(F.softmax(atty, dim=1))
            loss_adv_target1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
            loss_adv_target2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
            loss_adv_targetf = bce_loss(D_f, Variable(torch.FloatTensor(D_f.data.size()).fill_(source_label)).cuda())
            loss = loss_seg2 + LAMBDA_SEG * loss_seg1 + LAMBDA_ADV_TARGET1 * loss_adv_target1 + \
                   LAMBDA_ADV_TARGET2 * loss_adv_target2 + LAMBDA_ADV_DF * loss_adv_targetf + LAMBDA_MMD * loss_mmd
            #将总损失反向传播，以更新生成器模型的参数。
            loss.backward()

            # train D
            # bring back requires_grad
            #将判别器模型的参数设置为需要梯度，以允许在训练判别器时更新其参数。

            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True
            for param in model_Df.parameters():
                param.requires_grad = True
            # 分别使用源域和目标域的预测结果，计算判别器的损失，并进行反向传播。
            # train with source
            #这些操作将预测结果从计算图中分离出来，使得在计算判别器的损失时不会影响到生成器模型的参数
            pred1 = pred1.detach()
            pred2 = pred2.detach()
            attx = attx.detach()
            D_out1 = model_D1(F.softmax(pred1, dim=1))
            D_out2 = model_D2(F.softmax(pred2, dim=1))
            D_f = model_Df(F.softmax(attx, dim=1))
            #这些操作计算判别器的输出与期望标签（这里是源域标签）之间的二元交叉熵损失（Binary Cross Entropy Loss
            loss_D1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
            loss_D2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
            loss_Df = bce_loss(D_f, Variable(torch.FloatTensor(D_f.data.size()).fill_(source_label)).cuda())
            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2
            loss_D1.backward()
            loss_D2.backward()
            loss_Df.backward()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()
            atty = atty.detach()
            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))
            D_f = model_Df(F.softmax(atty, dim=1))
            loss_D1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())
            loss_D2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())
            loss_Df = bce_loss(D_f, Variable(torch.FloatTensor(D_f.data.size()).fill_(target_label)).cuda())
            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2
            loss_D1.backward()
            loss_D2.backward()
            loss_Df.backward()

            #使用优化器的 step 方法更新模型的参数。
            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_Df.step()

            #在每个迭代中，打印出训练的进度、学习率、损失值、准确率和时间消耗。
            #每隔一定数量的迭代，进行一次验证，并根据验证结果决定是否保存当前模型。
            if iter % 1 == 0:
                clear_output()
                pred = np.argmax(pred_target2.data.cpu().numpy()[0], axis=0)
                gt = labels_t.data.cpu().numpy()[0]
                end_time = time.time()
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)] lr: {:.12} lr_D: {:.12} Loss: {:.6} Loss_seg: {:.6} Loss_adv: {:.6} Loss_mmd: {:.6} Loss_D1: {:.6} Loss_D2: {:.6} Accuracy: {:.2f}% Timeuse: {:.2f}'.format(
                    epoch, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], optimizer_D1.state_dict()['param_groups'][0]['lr'],
                    loss.data, loss_seg2.data, loss_adv_target2.data, loss_mmd.data, loss_D1.data, loss_D2.data, accuracy(pred, gt), end_time - start_time))
                start_time = time.time()
            iter += 1
            del (images, labels, images_t, labels_t, loss, loss_D1, loss_D2)

            if iter % 100 == 0:
                # We validate with the largest possible stride for faster computing
                start_time = time.time()
                MIoU = test(test_ids_P, test_ids_V, all=False, stride=32)
                end_time = time.time()
                print('Test Stide_32 time use: ', end_time - start_time)
                start_time = time.time()
                if MIoU > MIoU_best:
                    torch.save(net.state_dict(), './Train_Model/DATrans_P2V_epoch{}_{}'.format(epoch, MIoU))
                    MIoU_best = MIoU
    print("Train Done!!")

train(50)

###  Test  ####
# net.load_state_dict(torch.load('Train_Model/'))
# start = time.time()
# acc, all_preds, all_gts = test(test_ids_P, test_ids_V, all=True, stride=32)
# print('Test tride time use: ', time.time() - start)
# print("Acc: ", acc)
# for p, id_ in zip(all_preds, test_ids_V):
#     img = convert_to_color(p)
#     io.imsave('./Test_Vision/P2V_tile_{}.png'.format(id_), img)