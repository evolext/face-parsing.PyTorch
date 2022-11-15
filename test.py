#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0 ],      # 0:  ?
        [255, 85, 0],      # 1:  skin
        [55, 170, 0],      # 2:  l_brow
        [255, 0, 85],      # 3:  r_brow
        [55, 0, 170],      # 4:  l_eye 
        [0, 255, 0],       # 5:  r_eye
        [85, 255, 0],      # 6:  ?
        [170, 255, 0],     # 7:  l_ear
        [0, 255, 85],      # 8:  r_ear
        [0, 255, 170],     # 9:  ?
        [0, 0, 255],       # 10: nose
        [85, 0, 255],      # 11: ?
        [170, 0, 255],     # 12: u_lip
        [0, 85, 255],      # 13: l_lip
        [0, 170, 255],     # 14: neck
        [255, 255, 0],     # 15: ?
        [255, 170, 255],   # 16: cloth
        [255, 255, 170],   # 17: hair
        [255, 0, 255],     # 18: hat (то, что определяется сверху)
    ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:

        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im


# Основная функция разбиения лица на сегменты
def evaluate_folder(input_path='./data', output_path='./result', model_path='./model/final.pth'):

    # Если директория для сохранения не существует, то будет создана
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Настройка модели
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for image_path in os.listdir(input_path):
            img = Image.open(os.path.join(input_path, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=os.path.join(output_path, image_path))


def evaluate_image(img, model_path='./model/final.pth'):

    # Настройка модели
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
        vis_img = vis_parsing_maps(image, parsing, stride=1, save_im=False)
        return vis_img, np.unique(parsing)





# def parse_folder(input_path='./test', output_path='./result'):
#     evaluate(dspth=input_path, respath=output_path)


#   0:  'background'
#   1:  'skin'
#   2:  'nose'
#   3:  'eye_g'
#   4:  'l_eye'
#   5:  'r_eye'
#   6:  'l_brow'
#   7:  'r_brow'
#   8:  'l_ear'
#   9:  'r_ear'
#   10: 'mouth'
#   11: 'u_lip'
#   12: 'l_lip'
#   13: 'hair'
#   14: 'hat'
#   15: 'ear_r'
#   16: 'neck_l'
#   17: 'neck'
#   18: 'cloth'

