#!/usr/bin/env python
# coding: utf-8

import os, sys
base = "/Users/ckh/Documents/Poladrone/nb"
# base = "D:/YoloV5_Hui/poladrone_nb"
sys.path.append(base + "/ref/pytorchYOLOv4")

from exp.nb_TrainingRunnner import *
from exp.nb_CustomDataLoader import Detection_dataset

from config.config import Cfg

import random
import torch

def custom_get_args():
    
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--batch-size', type=int,
                        help='Batch size', dest='batch')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-e', '--epoch', type=int,
                        help='Epoch', dest='TRAIN_EPOCHS')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default="-1",
                        help='GPU', dest='gpu')

    parser.add_argument('-pretrained',type=str,
                        help='pretrained yolov4.conv.137', dest='pretrained')
    
    parser.add_argument('-train_txt', type=str,
                        help="train dataset path (*.txt)", dest='train_label')
    parser.add_argument('-val_txt', type=str,
                        help="val dataset path (*.txt)", dest='val_label')
    
    parser.add_argument('-c', '--classes',type=int, 
                        help='dataset classes', dest='classes')

    # parser.add_argument(
    #     '-optimizer', type=str, default='adam',
    #     help='training optimizer',
    #     dest='TRAIN_OPTIMIZER')
    # parser.add_argument(
    #     '-iou-type', type=str, default='iou',
    #     help='iou type (iou, giou, diou, ciou)',
    #     dest='iou_type')
    # parser.add_argument(
    #     '-keep-checkpoint-max', type=int, default=10,
    #     help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
    #     dest='keep_checkpoint_max')

    args = vars(parser.parse_args())

    return args

if __name__ == "__main__":
    torch.manual_seed(20)
    np.random.seed(20)
    random.seed(20)

    cfg = Cfg
    args = custom_get_args()
    for k in args.keys():
        if args.get(k) is None: continue
        cfg[k] = args.get(k)
    
    # data_list = "D:/YoloV5_Hui/Dataset/train_npt.txt"
    # data_list = "/Users/ckh/OneDrive - Default Directory/Hui_Wan/train_npt.txt"
    data_list = cfg.train_label
    dataset = Detection_dataset(data_list, Cfg)

    val_data_list = cfg.val_label
    val_dataset = Detection_dataset(val_data_list, Cfg, val=True)

    logging = init_logger(log_dir='log', stdout=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    print("Using Device:", device)

    pretrained = cfg.get("pretrained", None)
    model = Yolov4(pretrained,n_classes=cfg.classes)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        custom_train( train_dataset=dataset, val_dataset=val_dataset,
            model=model, config=cfg,
            epochs=cfg.TRAIN_EPOCHS, device=device,
            log_step=100, val_epoch=10)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)