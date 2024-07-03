from backbone.model_irse import IR_SE_50
from backbone.model_resnet import ResNet_50
from config import configurations
import torch

from tensorboardX import SummaryWriter
from util.utils import buffer_val, get_val_data, perform_val

cfg = configurations[1]

SEED = cfg['SEED'] # random seed for reproduce results
torch.manual_seed(SEED)

DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train/val/test data are stored
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
NUM_EPOCH = cfg['NUM_EPOCH']
WEIGHT_DECAY = cfg['WEIGHT_DECAY']
MOMENTUM = cfg['MOMENTUM']
STAGES = cfg['STAGES'] # epoch stages to decay learning rate

DEVICE = cfg['DEVICE']
MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
GPU_ID = cfg['GPU_ID'] # specify your GPU ids
PIN_MEMORY = cfg['PIN_MEMORY']
NUM_WORKERS = cfg['NUM_WORKERS']

writer = SummaryWriter(LOG_ROOT)

STATE_DICT = torch.load("./model/Backbone_ResNet_50_Best.pth")

BACKBONE = ResNet_50(INPUT_SIZE)
BACKBONE.load_state_dict(STATE_DICT)
BACKBONE.to(DEVICE)

lfw, lfw_issame = get_val_data(DATA_ROOT)

accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, 15)
print("Evaluation: LFW Acc: {}".format(accuracy_lfw))