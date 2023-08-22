import torch
import os
import numpy as np
import json
from datetime import datetime
import argparse
import pandas as pd
from datasets.crowd_bee import Crowd
from models.vgg0 import vgg19
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default="/home/vuhai/Tung-Bayesian-Bee/dataBee_preprocessed_ver3/",
                        help='test data directory')
    parser.add_argument('--save-dir', default="/home/vuhai/Tung-Bayesian-Bee/logs_noise/r_13_0.2/0810-032316/",
                        help='model directory')
    parser.add_argument('--results-dir', default="/home/vuhai/Tung-Bayesian-Bee/results/Log_final/",
                      help='save outputs')
    parser.add_argument('--device', default='1', help='assign device')
    args = parser.parse_args()
    return args

def save_results(name_ls, grt_count_ls, pred_count_ls, mae, mse, data_path, yolo_path, save_path):

    #Save folder
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    first_dir = os.path.dirname(os.path.dirname(data_path))
    sub_folder = os.path.basename(first_dir)
    save_folder = os.path.join(save_path, sub_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #Get YOLO count
    yolo_count_ls = []
    for filetext in os.listdir(yolo_path):
        yolo_dir = os.path.join(yolo_path, filetext)
        with open(yolo_dir, 'r') as yf:
            lines = yf.readlines()
        yf.close()
        yolo_count_ls.append(len(lines))
    
    #Save count result
    result_dict = {}
    result_dict['Name'] = name_ls
    result_dict['Ground-truth count'] = grt_count_ls
    result_dict['Predict count'] = pred_count_ls
    result_dict['Yolo count'] = yolo_count_ls

    result_df = pd.DataFrame.from_dict(result_dict)
    result_df_name = sub_folder + "_result.csv"
    result_df.to_csv(os.path.join(save_folder, result_df_name))
    
    #Save mae mse
    mae_mse_name = sub_folder + "_mae_mse.txt"
    mae_mse_file = os.path.join(save_folder, mae_mse_name)
    with open(mae_mse_file, 'w') as f:
        f.write('MAE: ' + str(mae) + '\n')
        f.write('MSE: ' + str(mse) + '\n')
    f.close()
    

if __name__ == '__main__':
    
    yolo_path = "/home/vuhai/Tung-Bayesian-Bee/YOLO_results_test/"

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)
    model = vgg19()
    device = torch.device('cuda')
    model.to(device)
    
    print(model)
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))
    epoch_minus = []
    
    name_ls = []
    grt_count_ls = []
    pred_count_ls = []

    for inputs, count, name in dataloader:

        inputs = inputs.to(device)
        
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            
            grt_count = count[0].item()
            pred_count = round(torch.sum(outputs).item())
            
            temp_minu = grt_count - pred_count
            print(name, temp_minu, grt_count, round(torch.sum(outputs).item(), 2))
            
            epoch_minus.append(temp_minu)
            name_ls.append(str(name[0]))
            grt_count_ls.append(grt_count) 
            pred_count_ls.append(pred_count) 
    
    
    epoch_minus = np.array(epoch_minus)
    mse = round(np.sqrt(np.mean(np.square(epoch_minus))), 4)
    mae = round(np.mean(np.abs(epoch_minus)), 4)
    log_str = 'Final Test: mae {}, mse {}'.format(mae, mse)
    
    save_results(name_ls, grt_count_ls, pred_count_ls, mae, mse, args.save_dir, yolo_path, args.results_dir)
    
    print(log_str)

