import torch
import os
import numpy as np
from datasets.crowd_bee import Crowd_infer
from models.vgg import vgg19
import time
import pandas as pd
from utils.create_dir import create_dir



def get_count_per_day(day_root_dir, day_save_dir, model, device):
    print("--------------- START -----------------")
    count_day_dict = {}
    k_hours = []
    k_frames = []
    k_counts = []
    k_runtimes = []
    k_density_maps = []
    t0_day = time.time()

    for time_folder in os.listdir(day_root_dir):
        print("=====  Time: ", time_folder, "  =====")
        
        t0_time = time.time()
  
        #Directory to image folder
        image_dir = os.path.join(day_root_dir, time_folder)
        density_save_folder = os.path.join(day_save_dir, "density_maps")
        if not os.path.exists(density_save_folder):
            os.makedirs(density_save_folder)
        
        #Prepare dataset
        datasets = Crowd_infer(image_dir, 1024, 8, is_gray=False, method='val')
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)

        #Run inference
        for inputs, name in dataloader:
            t0_img = time.time()
            inputs = inputs.to(device)
            #print(inputs.shape)
            assert inputs.size(0) == 1, 'the batch size should equal to 1'
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                img_runtime = round(time.time() - t0_img, 2)
                img_count = round(torch.sum(outputs).item())
                dm = outputs.squeeze().detach().cpu().numpy()
                img_dm = dm / np.max(dm)

                img_frame = str(name[0]).split(" ")[1]
                #print(img_frame)
                img_hour = "_".join(time_folder.split("-")[-2:])
                #print(img_hour)

                k_hours.append(img_hour)
                k_frames.append(img_frame)
                k_counts.append(img_count)
                k_runtimes.append(img_runtime)
                k_density_maps.append(img_dm)

                #Save density maps
                dm_text = os.path.join(density_save_folder, str(name[0]) + ".npy")
                np.save(dm_text, img_dm)
                
        t1_time = round(time.time() - t0_time, 2)
        print("Video 514 frames het: ", str(t1_time), " s")
        


    #Save date csv
    count_day_dict["Hour"] = k_hours
    count_day_dict["Frame"] = k_frames
    count_day_dict["Count"] = k_counts
    count_day_dict["Runtime"] = k_runtimes
    count_day_dict["DensityMap"] = k_density_maps

    day_df = pd.DataFrame.from_dict(count_day_dict)
    datedf = os.path.basename(day_root_dir) + ".csv"
    day_df.to_csv(os.path.join(day_save_dir, datedf))
    
    t1_day = round(time.time()-t0_day, 2)
    print("Ca ngay : ", str(t1_day), " s")
    
    print("--------------- DONE -----------------")

    return count_day_dict


if __name__ == '__main__':
    
    #Example
    day_root_dir = "/home/vuhai/Tung-Bayesian-Bee/real_data/image_real_example/Ex_hive_5/Ex_2022/Ex_7-2022/Ex_09-07-2022"
    day_save_dir = "/home/vuhai/Tung-Bayesian-Bee/results/demo/09-07-2022"
    model_dir = "/home/vuhai/Tung-Bayesian-Bee/logs/Drop_0.2_0.2/0714-092244"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'.strip()  # set vis gpu
                                             
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth'), device))
    
    dt = get_count_per_day(day_root_dir, day_save_dir, model, device)
                                             
    