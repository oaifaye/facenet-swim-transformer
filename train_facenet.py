import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.facenet import Facenet
from nets.facenet_training import triplet_loss, LossHistory, weights_init
from utils.dataloader import FacenetDataset, dataset_collate
from utils.eval_metrics import evaluate
from utils.LFWdataset import LFWDataset
import argparse

def get_num_classes(train_txt_path, val_txt_path):
    with open(train_txt_path) as f:
        train_paths = f.readlines()
    with open(val_txt_path) as f:
        val_paths = f.readlines()
    labels = []
    for path in train_paths:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    for path in val_paths:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

def get_data_lines(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # num_val = int(len(lines) * val_split)
    # num_train = len(lines) - num_val
    return lines

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_ont_epoch(model,loss,epoch,epoch_size,gen,val_epoch_size,gen_val,max_poch,test_loader,cuda,batch_size):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0
    total_loss = 0

    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{max_poch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images  = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    labels  = Variable(torch.from_numpy(labels).long()).cuda()
                else:
                    images  = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    labels  = Variable(torch.from_numpy(labels).long())

            optimizer.zero_grad()
            before_normalize, outputs1  = model.forward_feature(images)
            _triplet_loss = loss(outputs1, batch_size)

            _CE_loss = 0
            if need_cbloss:
                outputs2                    = model.forward_classifier(before_normalize)
                _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2,dim=-1),labels)
            _loss           = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()

            if need_cbloss:
                with torch.no_grad():
                    accuracy         = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
                total_accuracy  += accuracy.item()
                total_CE_loss += _CE_loss.item()

            total_triple_loss   += _triplet_loss.item()
            total_loss       += _loss.item()
            pbar.set_postfix(**{'total_triple_loss' : total_triple_loss / (iteration + 1),
                                'total_CE_loss'     : total_CE_loss / (iteration + 1),
                                'accuracy'          : total_accuracy / (iteration + 1),
                                'total_loss'             : total_loss / (iteration + 1),
                                'lr'                : get_lr(optimizer)})

            pbar.update(1)

            if iteration > 0 and (iteration % save_iteration == 0 or iteration == len(gen) - 1):
                val_total_triple_loss = 0
                val_total_CE_loss = 0
                val_total_accuracy = 0
                val_total_loss = 0
                net.eval()
                print('Start Validation')
                with tqdm(total=val_epoch_size, desc=f'Epoch {epoch + 1}/{max_poch}',postfix=dict,mininterval=0.3) as pbar_val:
                    for iteration_val, batch_val in enumerate(gen_val):
                        if iteration_val >= val_epoch_size:
                            break
                        images_val, labels_val = batch_val
                        with torch.no_grad():
                            if cuda:
                                images_val  = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                                labels_val  = Variable(torch.from_numpy(labels_val).long()).cuda()
                            else:
                                images_val  = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                                labels_val  = Variable(torch.from_numpy(labels_val).long())

                            # optimizer.zero_grad()
                            before_normalize_val, outputs1_val  = model.forward_feature(images_val)
                            _triplet_loss_val   = loss(outputs1_val, batch_size)

                            _CE_loss_val = 0
                            if need_cbloss:
                                outputs2_val                    = model.forward_classifier(before_normalize_val)
                                _CE_loss_val        = nn.NLLLoss()(F.log_softmax(outputs2_val,dim=-1),labels_val)
                            _loss_val           = _triplet_loss_val + _CE_loss_val

                            if need_cbloss:
                                accuracy_val        = torch.mean((torch.argmax(F.softmax(outputs2_val, dim=-1), dim=-1) == labels_val).type(torch.FloatTensor))
                                val_total_accuracy  += accuracy_val.item()
                                val_total_CE_loss += _CE_loss_val.item()

                            val_total_triple_loss   += _triplet_loss_val.item()
                            val_total_loss       += _loss_val.item()

                        pbar_val.set_postfix(**{'val_total_triple_loss' : val_total_triple_loss / (iteration_val + 1),
                                            'val_total_CE_loss'     : val_total_CE_loss / (iteration_val + 1),
                                            'val_total_loss'     : val_total_loss / (iteration_val + 1),
                                            'val_accuracy'          : val_total_accuracy / (iteration_val + 1),
                                            'lr'                    : get_lr(optimizer)})
                        pbar_val.update(1)

                print("开始进行LFW数据集的验证。")
                labels_test, distances_test = [], []
                for _, (data_a, data_p, label_test) in enumerate(test_loader):
                    with torch.no_grad():
                        data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                        if cuda:
                            data_a, data_p = data_a.cuda(), data_p.cuda()
                        data_a, data_p, label_test = Variable(data_a), Variable(data_p), Variable(label_test)
                        out_a, out_p = model(data_a), model(data_p)
                        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                    distances_test.append(dists.data.cpu().numpy())
                    labels_test.append(label_test.data.cpu().numpy())

                labels_test = np.array([sublabel for label in labels_test for sublabel in label])
                distances_test = np.array([subdist for dist in distances_test for subdist in dist])
                _, _, accuracy_test, _, _, _, _ = evaluate(distances_test, labels_test)
                print('LFW_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy_test), np.std(accuracy_test)))

                loss_history.append_loss(np.mean(accuracy_test), (total_triple_loss+total_CE_loss)/(epoch_size+1), (val_total_triple_loss+val_total_CE_loss)/(val_epoch_size+1))
                print('Finish Validation')
                print('Epoch:' + str(epoch+1) + '/' + str(max_poch))
                print('Total Loss: %.4f' % ((total_triple_loss+total_CE_loss)/(iteration+1)))
                print('Saving state, iter:', str(epoch+1))
                torch.save(model.state_dict(), 'logs/Epoch%d-Iteration%d-Total_Loss%.4f.pth-Val_Loss%.4f-Accuracy%.4f.pth'%((epoch+1), iteration,
                                                                (total_triple_loss+total_CE_loss)/(iteration+1),
                                                                val_total_loss / (iteration_val + 1),
                                                                np.mean(accuracy_test)))
                net.train()
    
    return (val_total_triple_loss + val_total_CE_loss)/(val_epoch_size+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs/', type=str, help='断点文件存放目录')
    parser.add_argument('--train_txt_path', default='train_path.txt', type=str, help='训练图片路径的txt文件')
    parser.add_argument('--val_txt_path', default='val_path.txt', type=str, help='验证图片路径的txt文件')
    parser.add_argument('--lfw_dir', default='lfw/', type=str, help='LFW测试集目录')
    parser.add_argument('--lfw_txt_path', default='model_data/lfw_pair.txt', type=str, help='LFW测试集txt目录')
    parser.add_argument('--model_path', default='logs/Epoch1-Iteration35000-Total_Loss0.0826.pth-Val_Loss0.0852-Accuracy0.9882.pth', type=str, help='用于继续练的模型文件')
    parser.add_argument('--warmup_epoch', default=0, type=int, help='warmup的结束轮次，如果不使用warmup，传0')
    parser.add_argument('--max_epoch', default=20, type=int, help='训练最大轮次')
    parser.add_argument('--save_iteration', default=5000, type=int, help='每迭代多少次进行验证并存断点')
    parser.add_argument('--need_cbloss', default=False, type=bool, help='是否需要一个全连接分类器帮助收敛，一般从零开始训练需要True')
    parser.add_argument('--input_size', default=160, type=int, help='输入宽高')
    parser.add_argument('--lr_1', default=1e-3, type=float, help='warmup的起始LR，不使用warmup时，该参数没有作用')
    parser.add_argument('--batch_size_1', default=64, type=int, help='warmup的batch_size，不使用warmup时，该参数没有作用')
    parser.add_argument('--lr_2', default=1e-4, type=float, help='全参数训练的起始LR')
    parser.add_argument('--batch_size_2', default=128, type=int, help='全参数训练的batch_size')
    parser.add_argument('--num_workers', default=32, type=int, help='dataloader的线程数')
    parser.add_argument('--backbone', default='inception_resnetv1', type=str, help='选backbone,包括mobilenet/inception_resnetv1/sim-t')
    args = parser.parse_args()

    log_dir = args.log_dir
    train_txt_path = args.train_txt_path
    val_txt_path = args.val_txt_path
    # model_path = "model_data/facenet_inception_resnetv1.pth"
    model_path = args.model_path
    warmup_epoch = args.warmup_epoch
    max_epoch = args.max_epoch
    save_iteration = args.save_iteration
    # 是否需要一个全连接分类器帮助收敛，一般从零开始训练需要True
    need_cbloss = args.need_cbloss
    num_classes = get_num_classes(train_txt_path, val_txt_path)
    input_shape = [args.input_size, args.input_size, 3]
    lr_1 = args.lr_1
    batch_size_1 = args.batch_size_1
    lr_2 = args.lr_2
    batch_size_2 = args.batch_size_2
    lfw_dir = args.lfw_dir
    lfw_txt_path = args.lfw_txt_path
    num_workers = args.num_workers
    backbone = args.backbone
    print('args:', args)

    #--------------------------------------#
    #   Cuda的使用
    #--------------------------------------#
    use_cuda = torch.cuda.is_available()

    model = Facenet(num_classes=num_classes, backbone=backbone)
    weights_init(model)
    if os.path.exists(model_path):
        print('加载预训练模型：', model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    net = model.train()
    if use_cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    loss = triplet_loss(alpha=0.4)
    loss_history = LossHistory(log_dir)

    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir, pairs_path=lfw_txt_path, image_size=input_shape), batch_size=32, shuffle=False)

    train_data = get_data_lines(train_txt_path)
    val_data = get_data_lines(val_txt_path)
    train_dataset   = FacenetDataset(input_shape, train_data, num_classes)
    val_dataset     = FacenetDataset(input_shape, val_data, num_classes)

    # warmup阶段
    if warmup_epoch > 0:
        print('开始warmup')
        optimizer = optim.Adam(net.parameters(), lr_1)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_1, num_workers=num_workers, pin_memory=True,
                                  drop_last=True, collate_fn=dataset_collate)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_1, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        epoch_size = max(1, len(train_data) // batch_size_1)
        val_epoch_size = max(1, len(val_data) // batch_size_1)
        for param in model.backbone.parameters():
            param.requires_grad = False
        for epoch in range(0, warmup_epoch):
            _loss = fit_ont_epoch(model,loss,epoch,epoch_size,train_loader,val_epoch_size,val_loader,warmup_epoch,
                                  LFW_loader,use_cuda, batch_size_1)
            lr_scheduler.step(_loss)

    # 全参数训练
    print('开始全参数训练')
    optimizer = optim.Adam(net.parameters(), lr_2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4, verbose=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_2, num_workers=num_workers, pin_memory=True,
                              drop_last=True, collate_fn=dataset_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_2, num_workers=num_workers, pin_memory=True,
                            drop_last=True, collate_fn=dataset_collate)
    epoch_size = max(1, len(train_data) // batch_size_2)
    val_epoch_size = max(1, len(val_data) // batch_size_2)
    for param in model.backbone.parameters():
        param.requires_grad = True
    for epoch in range(warmup_epoch, max_epoch):
        _loss = fit_ont_epoch(model, loss, epoch, epoch_size, train_loader, val_epoch_size, val_loader, max_epoch,
                              LFW_loader, use_cuda, batch_size_2)
        lr_scheduler.step(_loss)