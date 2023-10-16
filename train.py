import os
import yaml
import torch
import argparse
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from data import get_dataset
import utils.utils as utils
from utils.utils import create_log_folder, model_info
from utils import function
from models.plateNet import myNet_ocr

import models.crnn as crnn
from models.LPRNet import build_lprnet

def parse_arg():
    parser = argparse.ArgumentParser(description="Train crnn")
    parser.add_argument('--cfg', default='configs/360CC_config.yaml', help='experiment configuration filename', type=str)
    parser.add_argument('--img_h', type=int, default=48, help='height') 
    parser.add_argument('--img_w', type=int, default=168, help='width')
    args = parser.parse_args()

    # Load cfg.yaml
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.MODEL.NUM_CLASSES = len(config.PLATENAME)
    config.HEIGHT = args.img_h
    config.WIDTH = args.img_w
    return config

def main():
    # Load config
    config = parse_arg()

    # Create output folder
    output_dict = create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK  # True
    cudnn.deterministic = config.CUDNN.DETERMINISTIC # Falses
    cudnn.enabled = config.CUDNN.ENABLED # True

    # Writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    #cfg =[8,8,16,16,'M',32,32,'M',48,48,'M',64,128] #small model
    cfg =[16,16,32,32,'M',64,64,'M',96,96,'M',128,256]#medium model
    # cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256] #big model

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    # Build model
    # model = crnn.get_crnn(config,cfg=cfg)
    model = myNet_ocr(num_classes=config.MODEL.NUM_CLASSES + 1, cfg=cfg) # +1 为 '#'
    # model = build_lprnet(num_classes=len(plate_chr))
    model = model.to(device)

    # Loss
    criterion = torch.nn.CTCLoss()

    # Optimizer
    optimizer = utils.get_optimizer(config, model)

    # Learning rate
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    # Pre-trained
    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        # from collections import OrderedDict
        # model_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     if 'cnn' in k:
        #         model_dict[k[4:]] = v
        # model.cnn.load_state_dict(model_dict)
        model.load_state_dict(checkpoint)
        # if config.TRAIN.FINETUNE.FREEZE:
        #     for p in model.cnn.parameters():
        #         p.requires_grad = False
    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)

    # DataLoader
    train_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=True)   # [img_path, car_plate '云A005HL']
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    val_dataset = get_dataset(config)(config,input_w=config.WIDTH, input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # ----------------------------------> Train and val <----------------------------------
    best_acc = 0.5
    converter = utils.strLabelConverter(config.PLATENAME)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # Train
        function.train(config, train_loader, train_dataset, converter, model,
                       criterion, optimizer, device, epoch, writer_dict, output_dict)
        lr_scheduler.step()
        # Val
        acc = function.validate(config, val_loader, val_dataset, converter,
                                model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "cfg":cfg,
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
