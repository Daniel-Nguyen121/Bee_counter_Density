from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            m_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio

def find_dis(point):
    square = np.sum(point*point, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    im_bee = Image.open(im_path) 
    im_w_bee, im_h_bee = im_bee.size  
    mat_pth_bee = im_path.replace('.jpg', '.txt')

    with open(mat_pth_bee, 'r') as f:
        lines = f.readlines()
  
    x_ls = []
    y_ls = []
    x_cor = []
    y_cor = []

    for line in lines:
        ls_coor = line.split(" ")
        x_ls.append(float(ls_coor[1]))
        y_ls.append(float(ls_coor[2]))

    num_bee = len(x_ls)
    for i in range(num_bee):
        e_cor_x = x_ls[i]*im_w_bee
        e_cor_y = y_ls[i]*im_h_bee
        x_cor.append(e_cor_x)
        y_cor.append(e_cor_y)

    coordinates = []
    for i in range(num_bee):
        coordinates.append([x_cor[i], y_cor[i]])
  
    points = np.asarray(coordinates).astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w_bee) * (points[:, 1] >= 0) * (points[:, 1] <= im_h_bee)
    # print(idx_mask)
    points = points[idx_mask]
    # print(points)
    im_h_bee, im_w_bee, rr = cal_new_size(im_h_bee, im_w_bee, min_size, max_size)
    im = np.array(im_bee)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w_bee, im_h_bee), cv2.INTER_CUBIC)
        points = points * rr
  
    return Image.fromarray(im), points

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default="/home/vuhai/Tung-Bayesian-Bee/Bee_comvis_ver_3_aug", help='original data directory')
    parser.add_argument('--data-dir', default="/home/vuhai/Tung-Bayesian-Bee/dataBee_preprocessed_ver3_aug", help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    origin_dir = args.origin_dir
    min_size = 512
    max_size = 2048
    
    for phase in ['Train', 'Test', 'Val']:
        sub_dir = os.path.join(origin_dir, phase)
        if phase == 'Train':
            sub_save_dir = os.path.join(save_dir, phase.lower())
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            
            print('Process Training data:\n')
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for file in im_list:
                im_path = file
                name = os.path.basename(file)
                print(name)
            
                im, points = generate_data(im_path)
                dis = find_dis(points)
                points = np.concatenate((points, dis), axis=1)
                
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)
            print('\n=========\n')
        elif phase == 'Val':
            sub_save_dir = os.path.join(save_dir, phase.lower())
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            
            print('Process Validation data:\n')
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for file in im_list:
                im_path = file
                name = os.path.basename(file)
                print(name)
                
                im, points = generate_data(im_path)
                
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)
            print('\n=========\n')
        else:
            sub_save_dir = os.path.join(save_dir, phase.lower())
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
            
            print('Process Test data:\n')
            im_list = glob(os.path.join(sub_dir, '*jpg'))
            for im_path in im_list:
                name = os.path.basename(im_path)
                print(name)
                
                im, points = generate_data(im_path)
                
                im_save_path = os.path.join(sub_save_dir, name)
                im.save(im_save_path)
                gd_save_path = im_save_path.replace('jpg', 'npy')
                np.save(gd_save_path, points)