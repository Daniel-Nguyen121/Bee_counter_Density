import torch
import os
import cv2
import numpy as np
import pandas as pd
import time
from PIL import Image




import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F

import torch.utils.data as data
from torch.utils.data import DataLoader

#import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), "..")) 
from models.vgg import vgg19
from datasets.crowd_bee import Crowd_infer_individual_notPath

def checking_video(video_path, min_size = 1.0):
    '''
    Check if video too small --> skip
    '''
    if not os.path.exists(video_path):
        return False
    file_size_mb = os.stat(video_path).st_size / (1024*1024)
    #print(file_size_mb)
    if file_size_mb < min_size:
        return False
    else:
        return True
        
def count_bee_on_video(video_path, save_path, model_path, device, fps=20):

    #For save results
        #Time
    running_time = {}
    total_time = []     #Time to predict all
    delta_0 = []        #Time to create image
    delta_1 = []        #Time to prepare dataloader
    delta_2 = []        #Time to predict bee

        #For detailed results
    count_vid_dict = {}
    k_hours = []
    k_frames = []
    k_counts = []
    k_runtimes = []
    k_density_maps = []


    #Set environment and GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = device.strip()

    total_time_0 = time.time()

    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_path), device))

    if not checking_video(video_path):
        print('Video too small, can not process!')
        name_vid = os.path.basename(video_path).replace('.mp4', ' ')
        img_hour = "_".join(name_vid.split("-")[-2:])
        k_hours.append(img_hour)
        k_frames = [0]
        k_counts = [0]
        k_runtimes = [0]
        k_density_maps = [0]

        save_dir = os.path.join(save_path, name_vid)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Saving results: " + name_vid + " is: 0 bee")
    else:
        #Cut video
        cap = cv2.VideoCapture(video_path)
        frame_skip = fps
        frame_count = 0
        i = 0
        name_vid = os.path.basename(video_path)
        save_dir = os.path.join(save_path, name_vid.replace('.mp4', ' '))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("Process Video: " + name_vid)

        while cap.isOpened():
            #reading from frame
            ret, frame = cap.read()
            if not ret:
                break
            if i > frame_skip - 1:

                #Create an image
                t_0 = time.time()
                frame_count += 1
                # if video is still left continue creating images
                v_name = name_vid.replace('.mp4', ' ') + str(frame_count) + '.jpg'
                #output_path = os.path.join(output_dir, name)
                #print ('Creating...' + v_name)

                # writing the extracted images and resize from (1920, 1080) to (960, 540)
                resize = cv2.resize(frame, (960, 540), interpolation = cv2.INTER_LINEAR)
                #cv2.imwrite(output_path, resize)
                m_0 = round(time.time()-t_0, 4)
                delta_0.append(m_0)
                #print('Create an image takes: '+str(m_0)+' s')

                #Prepare dataset
                t_1 = time.time()
                datasets = Crowd_infer_individual_notPath(resize, v_name)
                dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)
                m_1 = round(time.time()-t_1, 4)
                delta_1.append(m_1)
                #print('Dataloader takes: '+str(m_1)+ ' s')

                #Inference
                for inputs, name in dataloader:
                    t_2 = time.time()
                    inputs = inputs.to(device)
                    assert inputs.size(0) == 1, 'the batch size should equal to 1'
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        img_count = round(torch.sum(outputs).item())
                        dm = outputs.squeeze().detach().cpu().numpy()
                        img_dm = dm / np.max(dm)
                        m_2 = round(time.time()-t_2, 4)
                        delta_2.append(m_2)
                        print(name[0], ' : ', str(img_count))
                        #print('Counting takes: '+str(m_1)+ ' s')
                        #print('==========')

                        time_ls = str(name[0]).split(" ")
                        img_frame = time_ls[1]
                        #print(img_frame)
                        img_hour = "_".join(time_ls[0].split("-")[-2:])
                        #print(img_hour)

                        k_hours.append(img_hour)
                        k_frames.append(img_frame)
                        k_counts.append(img_count)
                        k_runtimes.append(m_2)
                        k_density_maps.append(img_dm)
                i = 0
                continue
            i += 1

        cap.release()
        cv2.destroyAllWindows()

        total_time_1 = round(time.time()-total_time_0, 2)
        print('C? video h?t: ' + str(total_time_1) + ' s')

    #Save date csv
    count_vid_dict["Hour"] = k_hours
    count_vid_dict["Frame"] = k_frames
    count_vid_dict["Count"] = k_counts
    count_vid_dict["Runtime"] = k_runtimes
    count_vid_dict["DensityMap"] = k_density_maps

    day_df = pd.DataFrame.from_dict(count_vid_dict)
    day_df_name = name_vid + ".csv"
    day_df.to_csv(os.path.join(save_dir, day_df_name))

    running_time['Create image'] = delta_0
    running_time['Prepare dataloader'] = delta_1
    running_time['counting bee'] = delta_2
    time_df = pd.DataFrame.from_dict(running_time)
    time_df_name = name_vid + "_time.csv"
    time_df.to_csv(os.path.join(save_dir, time_df_name))

    print("========== DONE =========")

def count_bee_for_day(day_folder_path, save_day_path, model_path, device, fps=20):
    print('Process day: ' + os.path.basename(day_folder_path))
    save_path = os.path.join(save_day_path, os.path.basename(day_folder_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for video_file in sorted(os.listdir(day_folder_path)):
        video_path = os.path.join(day_folder_path, video_file)
        video_name = video_file.replace('.mp4', ' ')
        count_bee_on_video(video_path, save_path, model_path, device, fps)
    print('DONE!')

def count_bee_for_month(month_folder_path, save_month_path, model_path, device, fps=20):
    print('process month: ' + os.path.basename(month_folder_path))
    save_day_path = os.path.join(save_month_path, os.path.basename(month_folder_path))
    if not os.path.exists(save_day_path):
        os.makedirs(save_day_path)
    for day_folder in sorted(os.listdir(month_folder_path)):
        day_folder_path = os.path.join(month_folder_path, day_folder)
        count_bee_for_day(day_folder_path, save_day_path, model_path, device, fps)
    print('DONE!')

if __name__=="__main__":
  day_folder_path = "/home/vuhai/Tung-Bayesian-Bee/video_2_7/"
  save_day_path = "/home/vuhai/Tung-Bayesian-Bee/bayes_2_7/"
  model_path = "/home/vuhai/Tung-Bayesian-Bee/best_model_0.2.pth"
  device = '0'
  fp= 0
  
  count_bee_for_day(day_folder_path, save_day_path, model_path, device, 0)
  
  