import torch
import os
import numpy as np
from datasets.crowd_bee import Crowd_infer
from models.vgg import vgg19
import time
import pandas as pd
from utils.create_dir import create_dir
from utils.get_count_per_day import get_count_per_day

def get_count_for_hive(input_root_dir, output_root_dir, model, device):
    print("--------------- START -----------------")
    
    #Hive_level
    for hive_folder in os.listdir(input_root_dir):
        print("=====  Hive: ", hive_folder, "  =====")
        hive_root_dir, hive_save_dir = create_dir(input_root_dir, output_root_dir, hive_folder)
        hive_dict = {}

        #Year level
        for year_folder in os.listdir(hive_root_dir):
            print("==========  Year: ", year_folder, "  ==========")
            year_root_dir, year_save_dir = create_dir(hive_root_dir, hive_save_dir, year_folder)
            year_dict = {}

            #Month level
            for month_folder in os.listdir(year_root_dir):
                print("===============  Month: ", month_folder, "  ===============")
                month_root_dir, month_save_dir = create_dir(year_root_dir, year_save_dir, month_folder)
                count_month_dict = {}

                #Day level
                for day_folder in os.listdir(month_root_dir):
                    print("====================  Day: ", day_folder, "  ====================")
                    day_root_dir, day_save_dir = create_dir(month_root_dir, month_save_dir, day_folder)

                    count_day_dict = get_count_per_day(day_root_dir, day_save_dir, model, device)
                    #To month
                    count_month_dict[day_folder] = count_day_dict
                
                month_df = pd.DataFrame.from_dict(count_month_dict)
                month_name = month_folder + ".csv"
                month_df.to_csv(os.path.join(month_save_dir, month_name))

                #To year
                year_dict[month_folder] = count_month_dict
            year_df = pd.DataFrame.from_dict(year_dict)    
            year_name = year_folder + ".csv"
            year_df.to_csv(os.path.join(year_save_dir, year_name))

            #To hive
            hive_dict[year_folder] = year_dict
        hive_df = pd.DataFrame.from_dict(hive_dict)    
        hive_name = hive_folder + ".csv"
        hive_df.to_csv(os.path.join(hive_save_dir, hive_name))

    print("--------------- DONE -----------------")






if __name__ == '__main__':
    
    #Example
    input_dir = "/home/vuhai/Tung-Bayesian-Bee/real_data/image_real/"
    output_dir = "/home/vuhai/Tung-Bayesian-Bee/results/demo"
    model_dir = "/home/vuhai/Tung-Bayesian-Bee/logs/Drop_0.2_0.2/0714-092244/"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'.strip()  # set vis gpu
                                             
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth'), device))
    
    get_count_for_hive(input_dir, output_dir, model, device)
                                             
    