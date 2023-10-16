import torch
import cv2
import numpy as np
import os
import argparse
import yaml
from tqdm import tqdm
from easydict import EasyDict as edict

from models.plateNet import myNet_ocr


def image_processing(img_path, img_size, device):
    bgr_image = cv2.imread(img_path)
    # 将图像转换为RGB格式
    # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    img_h, img_w= img_size
    img = cv2.resize(bgr_image, (img_w,img_h))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def init_model(model_path, device):
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr(num_classes=len(plate_chr), export=True, cfg=cfg)  # export=True 用来推理
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def decodePlate(preds):
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    return newPreds

def get_plate_result(img, model):
    preds = model(img)
    preds = preds.argmax(dim=2)
    # print(preds)
    preds = preds.view(-1).detach().cpu().numpy()
    newPreds = decodePlate(preds)
    plate = ""
    for i in newPreds:
        plate += plate_chr[int(i)]
    return preds, plate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="测试某个文件夹下的预测准确度, accuracy")
    parser.add_argument('--model_path', type=str, default='output/360CC/crnn/2023-10-14-22-14/checkpoints/checkpoint_4_acc_0.9017.pth', help='model.pt path(s)')
    parser.add_argument('--cfg', default='configs/360CC_config.yaml', help='experiment configuration filename', type=str)
    parser.add_argument('--image_path', type=str, default='datasets/val', help='img path -> single image or file')
    opt = parser.parse_args()

    # Load cfg.yaml
    with open(opt.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    plate_chr = '#' + config.PLATENAME
    mean_value, std_value = config.DATASET.MEAN, config.DATASET.STD
    img_size = (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cpu")
    model = init_model(opt.model_path, device)

    right = 0
    all_files = os.listdir(opt.image_path)
    image_files = [os.path.join(opt.image_path, file) for file in all_files if
                   file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    for img_path in tqdm(image_files):
        img = image_processing(img_path, img_size, device)
        _, plate_pre = get_plate_result(img, model)

        plate_gt = img_path.split('/')[-1].split('\\')[-1].split('_')[0]    # "云A0E9H2_0.jpg"
        if (plate_pre == plate_gt):
            right += 1
        else:
            print(img_path, "\t", plate_gt, " rec as ---> ", plate_pre)

    print("Sum: %d, Right: %d, Accuracy: %f" % (len(image_files), right, right / len(image_files)))

