import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, random_split
import numpy as np
import matplotlib.pyplot as plt

from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from data.casia_datasets import CasiaDatasets
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from models.retinaface import RetinaFace
from util.utils import make_weights_for_balanced_classes, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.sampler import WeightedRandomSampler
# from data import dataset
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import os
import matplotlib.gridspec as gridspec

def train_val(inputs, labels, BACKBONE, HEAD, LOSS, DEVICE, losses, top1, top5):
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE).long()
    features = BACKBONE(inputs)
    outputs = HEAD(features, labels)
    loss = LOSS(outputs, labels)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
    losses.update(loss.data.item(), inputs.size(0))
    top1.update(prec1.data.item(), inputs.size(0))
    top5.update(prec5.data.item(), inputs.size(0))

    return loss, losses, top1, top5

parser = argparse.ArgumentParser(description='Train Loss Function')

parser.add_argument("-c", "--config", help = "configuration number", default = 2, type = int)

args = parser.parse_args()
if __name__ == '__main__':

    #======= hyperparameters & data loaders =======#
    cfg = configurations[args.config]

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
    # DATA_LIST_FILE = cfg['DATA_LIST_FILE']
    MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    OPTIMIZER_NAME = cfg['OPTIMIZER_NAME']
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    WARM_UP = NUM_EPOCH * .25
    STAGES = [int(NUM_EPOCH * .35), int(NUM_EPOCH * .65), int(NUM_EPOCH * .95)] # epoch stages to decay learning rate
    Total_Epoch = int(NUM_EPOCH + WARM_UP)
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    # Transformasi untuk data pelatihan
    train_transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    # Transformasi untuk data validasi
    val_transform = transforms.Compose([
        transforms.Resize([INPUT_SIZE[0], INPUT_SIZE[1]]), # smaller side resized
        # transforms.CenterCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    # Buat dataset penuh tanpa transformasi (untuk mendapatkan indeks dengan benar)
    full_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, 'casia-align-112'))
    validation_split = 0.2
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    train, val = random_split(full_dataset, [train_size, val_size])
    
    train_dataset = CasiaDatasets(train, transform=train_transform)
    val_dataset = CasiaDatasets(val, transform=val_transform)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)

    print("=" * 60)
    print("Validation DataLoader Generated")
    print("=" * 60)

    # Buat bobot untuk WeightedRandomSampler
    # weights = make_weights_for_balanced_classes(full_dataset, len(full_dataset.classes) if isinstance(full_dataset.classes, list) else full_dataset.classes)
    # np.savetxt("weights-cleaned.txt", weights)
    train_indices = train.indices
    weights = np.loadtxt("weights-cleaned.txt")
    weights = torch.DoubleTensor(weights)

    # Buat sampler untuk pelatihan dengan weighted random sampler pada subset pelatihan
    train_weights = weights[train_indices]  # Ambil bobot hanya untuk indeks pelatihan
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # Load data pelatihan
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST,
    )

    print("=" * 60)
    print("Training DataLoader Generated")
    print("=" * 60)

    NUM_CLASS = len(full_dataset.classes) if isinstance(full_dataset.classes, list) else full_dataset.classes
    print("Number of Training Classes: {}".format(NUM_CLASS))

    


    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE),
                    'ResNet_101': ResNet_101(INPUT_SIZE),
                    'ResNet_152': ResNet_152(INPUT_SIZE),
                    'IR_50': IR_50(INPUT_SIZE),
                    'IR_101': IR_101(INPUT_SIZE),
                    'IR_152': IR_152(INPUT_SIZE),
                    'IR_SE_50': IR_SE_50(INPUT_SIZE),
                    'IR_SE_101': IR_SE_101(INPUT_SIZE),
                    'IR_SE_152': IR_SE_152(INPUT_SIZE),
                    }
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    print("=" * 60)
    # print(BACKBONE)
    print("{} Backbone Generated".format(BACKBONE_NAME))
    print("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
                'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)}
    HEAD = HEAD_DICT[HEAD_NAME]
    print("=" * 60)
    print(HEAD)
    print("{} Head Generated".format(HEAD_NAME))
    print("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(),
                'Softmax': nn.CrossEntropyLoss(),
                }
    LOSS = LOSS_DICT[LOSS_NAME]
    print("=" * 60)
    print(LOSS)
    print("{} Loss Generated".format(LOSS_NAME))
    print("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
    
    OPTIMIZER_DICT = {'SGD': optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM),
                    'Adam': optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, betas = (0.9, 0.999), eps = 1e-8),
                    'Adamax': optim.Adamax([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, betas = (0.9, 0.999), eps = 1e-8),
                    }
    OPTIMIZER = OPTIMIZER_DICT[OPTIMIZER_NAME]
    # OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': backbone_paras_only_bn}], lr = LR, momentum = MOMENTUM)
    print("=" * 60)
    print(OPTIMIZER)
    print("Optimizer Generated")
    print("=" * 60)

    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
        HEAD = nn.DataParallel(HEAD, device_ids = GPU_ID)
        HEAD = HEAD.to(DEVICE)
    else:
        # single-GPU setting
        BACKBONE = BACKBONE.to(DEVICE)
        HEAD = HEAD.to(DEVICE)

    NUM_EPOCH_WARM_UP = Total_Epoch // WARM_UP  # use the first 1/25 epochs to warm up
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    batch = 0  # batch index

    best_acc = 0.0  # initialize with a small value
    top1_list_train = []  # buffer for accuracy values
    top5_list_train = []  # buffer for top5 accuracy values
    loss_list_train = []  # buffer for loss values
    top1_list_val = []  # buffer for accuracy values
    top5_list_val = []  # buffer for top5 accuracy values
    loss_list_val = []  # buffer for loss values
    best_epoch = 0  # best epoch index

    tqdm_epoch = tqdm(range(Total_Epoch), desc = "Epoch", leave=False, total=Total_Epoch)

    scaler = GradScaler()

    path = f"{MODEL_ROOT}/metric{HEAD_NAME}_backbone{BACKBONE_NAME}_lr{LR}_batch{BATCH_SIZE}_epoch{Total_Epoch}_opt{OPTIMIZER.__class__.__name__}"
    os.makedirs(path, exist_ok=True)
    curve_path = os.path.join(path, "Training_Loss_Accuracy.png")
    weights_path = os.path.join(path, "weights.pth")
    
    start = time.time()

    for epoch in tqdm_epoch: # start training process

        if epoch == STAGES[0]: # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == STAGES[2]:
            schedule_lr(OPTIMIZER)

        BACKBONE.train()  # set to training mode
        HEAD.train()

        losses_train = AverageMeter()
        top1_train = AverageMeter()
        top5_train = AverageMeter()
        losses_val = AverageMeter()
        top1_val = AverageMeter()
        top5_val = AverageMeter()
        tqdm_train = tqdm(train_loader, leave = False, total=len(train_loader))

        for inputs, labels in tqdm_train: # start to iterate over data_loader (batch training steps

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP): # adjust LR for each training batch during warm up
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

            loss, losses_train, top1_train, top5_train = train_val(inputs, labels,
                                                BACKBONE, HEAD, LOSS, DEVICE,
                                                losses_train, top1_train, top5_train)

            # compute gradient and do SGD step
            scaler.scale(loss).backward()
            scaler.step(OPTIMIZER)
            scaler.update()
            OPTIMIZER.zero_grad()

            # dispaly training loss & acc every DISP_FREQ
            # if (((batch + 1) % DISP_FREQ == 0) and batch != 0):
            #     print('Epoch {}/{} Batch {}/{}\t'
            #         'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #         'Training Prec@1 {top1_train.val:.3f} ({top1_train.avg:.3f})\t'
            #         'Training Prec@5 {top5_train.val:.3f} ({top5_train.avg:.3f})'.format(
            #         epoch + 1, NUM_EPOCH, batch + 1, len(train_loader) * NUM_EPOCH, loss = losses, top1_train = top1_train, top5_train = top5_train))
            tqdm_train.set_description("Epoch: {}/{} Loss: {:.4f} Prec@1: {:.3f} Prec@5: {:.3f}".format(epoch + 1, Total_Epoch, losses_train.avg, top1_train.avg, top5_train.avg))

            batch += 1 # batch index
        tqdm_eval = tqdm(val_loader, leave = False, total=len(val_loader))
        BACKBONE.eval()  # set to evaluation mode
        HEAD.eval()
        with torch.no_grad():
            for inputs, labels in tqdm_eval:
                _, losses_val, top1_val, top5_val = train_val(inputs, labels,
                                                    BACKBONE, HEAD, LOSS, DEVICE,
                                                    losses_val, top1_val, top5_val)
                tqdm_eval.set_description("Epoch: {}/{} Loss: {:.4f} Prec@1: {:.3f} Prec@5: {:.3f}".format(epoch + 1, Total_Epoch, losses_val.avg, top1_val.avg, top5_val.avg))

        # training statistics per epoch (buffer for visualization)
        top1_list_train.append(top1_train.avg)
        top5_list_train.append(top5_train.avg)
        loss_list_train.append(losses_train.avg)
        top1_list_val.append(top1_val.avg)
        top5_list_val.append(top5_val.avg)
        loss_list_val.append(losses_val.avg)
        
        np.savetxt(os.path.join(path, "Top1_Train.txt"),top1_list_train)
        np.savetxt(os.path.join(path, "Top5_Train.txt"),top5_list_train)
        np.savetxt(os.path.join(path, "Loss_Train.txt"),loss_list_train)
        np.savetxt(os.path.join(path, "Top1_Val.txt"),top1_list_val)
        np.savetxt(os.path.join(path, "Top5_Val.txt"),top5_list_val)
        np.savetxt(os.path.join(path, "Loss_Val.txt"),loss_list_val)
        
        fig = plt.figure(figsize=(10, 10))
        ax0 = fig.add_subplot(222, title="Top1 Training and Validation")
        ax1 = fig.add_subplot(221, title="Top5 Training and Validation")
        ax2 = fig.add_subplot(212, title="Loss Training and Validation")
        ax0.plot(top1_list_train, label="Training")
        ax0.plot(top1_list_val, label="Validation")
        ax1.plot(top5_list_train, label="Training")
        ax1.plot(top5_list_val, label="Validation")
        ax2.plot(loss_list_train, label="Training")
        ax2.plot(loss_list_val, label="Validation")
        ax0.legend()
        ax1.legend()
        ax2.legend()
        plt.savefig(os.path.join(path, "Training_Loss_Accuracy.png"))
        plt.close(fig)

        tqdm_epoch.set_description("Epoch: {}/{} Loss Train: {:.4f} Loss Val: {:.4f} @1 Train: {:.3f} @1 Val: {:.3f} @5 Train: {:.3f} @5 Val: {:.3f}".format(epoch + 1, Total_Epoch, losses_train.avg, losses_val.avg, top1_train.avg, top1_val.avg, top5_train.avg, top5_val.avg))

        if top1_val.avg > best_acc: # save best model
            best_acc = top1_val.avg
            # if MULTI_GPU:
            #     torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            #     torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
            # else:
            #     torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            #     torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
            torch.save(BACKBONE.state_dict(), os.path.join(path, "Backbone_{}_Best.pth".format(BACKBONE_NAME)))
            torch.save(HEAD.state_dict(), os.path.join(path, "Head_{}_Best.pth".format(HEAD_NAME)))
    hours, rem = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rem, 60)
    np.savetxt(os.path.join(path, "Training_Time.txt"), [hours, minutes, seconds])
