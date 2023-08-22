import torch
import os
import numpy as np
from datasets.crowd_bee import Crowd_infer
from models.vgg import vgg19
import argparse
import matplotlib.pyplot as plt
import time
import json
from utils.create_dir import create_dir
from utils.get_count_per_day import get_count_per_day
from utils.cut_frames_from_video.py

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--data-dir', default='/home/vuhai/Tung-Bayesian-Bee/real_data/image_real/',
                        help='training data directory')
    parser.add_argument('--model-dir', default='/home/vuhai/Tung-Bayesian-Bee/logs/Drop_0.2_0.2/0714-092244',
                        help='model directory')
    parser.add_argument('--save-dir', default='/home/vuhai/Tung-Bayesian-Bee/results/', help='save outputs')
    
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--model', type=str, default='vgg19', help='the model use to test')
    
    args = parser.parse_args()
    return args


def get_count_all_day(input_root_dir, output_root_dir, model, specified_folder=""):
    print("--------------- START -----------------")
    #model = vgg19()
    #device = torch.device('cuda')
    #model.to(device)
    #model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth'), device))
    
    if specified_folder != "":
        print("None")
    else:
        #Save count result for a month
        count_month = {}
        
        #For each day in month
        for sub_folder in os.listdir(input_root_dir):
            
            print("Day: ", str(sub_folder))
            
            sub_root_dir = os.path.join(input_root_dir, sub_folder)
            sub_save_dir = os.path.join(output_root_dir, sub_folder)
            if not os.path.exists(sub_save_dir):
                os.makedirs(sub_save_dir)
                
            #Save count result for a day
            count_day = {}
            
            #For each hour in day
            for filename in os.listdir(sub_root_dir):
                
                print("Process: ", str(filename))
                
                #Save count result for a time
                count_times = {}
                
                #Directory to image folder
                image_dir = os.path.join(sub_root_dir, filename)
        
                #Prepare dataset
                datasets = Crowd_infer(image_dir, 1024, 8, is_gray=False, method='val')
                dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False, num_workers=8, pin_memory=False)
                
                
                text_file = filename+ ".txt"
                image_save = os.path.join(sub_save_dir, filename)
                if not os.path.exists(image_save):
                    os.makedirs(image_save)
                
                count_day = []
                time_day = []
                
                
                with open(os.path.join(image_save, text_file), "w") as f:
                    for inputs, name in dataloader:
                        dict_times = {}
                        id_name = str(name[0]).split(" ")[1]
                        
                        t0_image = time.time()
                        inputs = inputs.to(device)
                        assert inputs.size(0) == 1, 'the batch size should equal to 1'
                        with torch.set_grad_enabled(False):
                            outputs = model(inputs)
                            image_count = round(torch.sum(outputs).item())
                            
                            count_day.append(image_count)
                            t1_image = round(time.time()-t0_image, 2)
                            time_day.append(t1_image)
                            
                            show_data = str(name[0]) + ":" + str(image_count)
                            print(show_data)
                            f.write(show_data+"\n")

                            dm = outputs.squeeze().detach().cpu().numpy()
                            dm_normalized = dm / np.max(dm)
                            dm_text = os.path.join(image_save, str(name[0]) + ".npy")
                            #print(dm_normalized.shape)
                            np.save(dm_text, dm_normalized)
                            
                            dict_times[id_name] = {"time": t1_image, "count":image_count, "dm_map":dm_normalized}
                    print("Len dict_times: " + str(len(dict_times)))
                    
                f.close() 
                dict_days[filename] = dict_times
                dict_all_folders[filename]=dict_days
        file_all = os.path.join(output_root_dir, "results_all.json")
        with open(file_all, "w") as fa:
            json.dump(dict_all_folders, fa)



if __name__ == '__main__':
    
    st = time.time()
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    #datasets = Crowd_infer(args.data_dir, 1024, 8, is_gray=False, method='val')
    #dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             #num_workers=8, pin_memory=False)
                                             
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    
    get_count_all_day(args.data_dir, args.results_dir, model, specified_folder="")
                                             
    