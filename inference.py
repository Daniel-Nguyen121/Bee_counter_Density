import torch
import os
import numpy as np
from datasets.crowd_bee import Crowd_infer
from models.vgg import vgg19
import argparse
import matplotlib.pyplot as plt
import time

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/vuhai/Tung-Bayesian-Bee/real_data/Out_real/',
                        help='training data directory')
    parser.add_argument('--save-dir', default='/home/vuhai/Tung-Bayesian-Bee/logs/Drop_0.2_0.2/0714-092244',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--model', type=str, default='vgg19',
                        help='the model use to test')
    parser.add_argument('--need-map', action='store_true',
                        help='whether draw density map')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    st = time.time()
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    #datasets = Crowd(os.path.join(args.data_dir, 'test'), 512, 8, is_gray=False, method='val')
    datasets = Crowd_infer(args.data_dir, 1024, 8, is_gray=False, method='val')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=8, pin_memory=False)

    #if args.model == 'vgg19':
    #    model = models.vgg.vgg19()
    #elif args.model == 'vgg16':
    #    model = models.vgg.vgg16()
    #elif args.model == 'resnet18':
    #    model = models.vgg.resnet18()
    #else:
    #    print("Invalid Model Type!")
    #    exit(0)
    
    model = vgg19()
    epoch_minus = []

    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), device))

    i = 0
    print("loading time: ", time.time() - st)
    tt = time.time()
    for inputs, name in dataloader:
        t0 = time.time()
        inputs = inputs.to(device)
        
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):

            outputs = model(inputs)
            
            #Cach 1:
            #sum_t1 = time.time()
            #sum = torch.sum(outputs)
            #sum = outputs.sum()
            #sum_t2 = time.time()
            #print(sum)
            #print(type(sum))
            #print(sum.shape)
            
            #item_t1 = time.time()
            #item = sum.item()
            #item = sum.item()
            #item_t2 = time.time()
            
            #round_t1 = time.time()
            #count = 0
            #count = round(item)
            #round_t2 = time.time()
            #count = round(torch.sum(outputs).item())
            
            #Cach 2:
            #count = outputs.sum().item()
            
            #round_time = time.time()-t4
            #count_time = time.time()-sum_t1
            #print(name, " : ", str(count))
            #sum_time = sum_t2-sum_t1
            #item_time = item_t2-item_t1
            #round_time = round_t2-round_t1
            
            #Cach 3
            dm = outputs.squeeze().detach().cpu().numpy()
            t = np.sum(dm)
            
            t1 = time.time()
            total = round(t1-t0, 2)
            print("This img takes: ", str(total), " s")
            #print("Sum: ", str(sum_time), " s, by: ", str(round(sum_time/total*100,2)), " %")
            #print("Item: ", str(item_time), " s, by: ", str(round(item_time/total*100,2)), " %")
            #print("Round: ", str(round_time), " s, by: ", str(round(round_time/total*100,2)), " %")
            #print("Count: ", str(count_time), " s, by: ", str(round(count_time/total*100,2)), " %")
            print("=============")    
            
            #tm = time.time()
            #dm = outputs.squeeze().detach().cpu().numpy()
            #print(type(dm))
            #tm = time.time()
            #t = np.vectorize(lambda x: x.sum())(tm)
            #t = np.sum(dm)
            #print(t)
            #print("Not   : ", str(round(time.time()-tm,2)), " s")
            #print("=============")

            if args.need_map:
                tm = time.time()
                dm = outputs.squeeze().detach().cpu().numpy()
                dm_normalized = dm / np.max(dm)
                plt.imshow(dm_normalized, cmap=plt.cm.jet, vmin=0, vmax=1)
                i += 1
                plt.savefig("./image/{}.png".format(i))
                
                
    td = time.time()
    print("Total: ", str(round(td-tt, 2)), " s")
    print("Done!!")