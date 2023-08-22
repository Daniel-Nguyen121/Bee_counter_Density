import numpy as np
#from scipy.stats import norm
#from scipy.special import ndtri
import cv2
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

##################################################
#Help functions
##################################################

def read_data_bee(img_path, txt_path):
    #convert img color space
    img_cvt = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    #read ground truth
    with open(txt_path, "r") as f:
      text = f.readlines()
  
    dt_coor = dict()
    x_ls = []
    y_ls = []
    a_ls = []
    b_ls = []
    for line in text:
      ls_coor = line.split(" ")
      x_ls.append(float(ls_coor[1]))
      y_ls.append(float(ls_coor[2]))
      a_ls.append(float(ls_coor[3]))
      b_ls.append(float(ls_coor[4]))
  
    dt_coor.update({"x": x_ls})
    dt_coor.update({"y": y_ls})
    dt_coor.update({"a": a_ls})
    dt_coor.update({"b": b_ls})
  
    return img_cvt, dt_coor

def norm_val(val):
    e_val = np.random.default_rng().random()
    if val >= 1.0:
        val = round(1.0 - e_val, 6)
    if val <= 0:
        val = round(e_val, 6)
    val = round(val, 6)
    return val

def gen_random_int_list(max_val, len_ls):
    random_ls = []
    for _ in range(len_ls):
        while True:
            random_int = random.randint(0, max_val-1)
            if random_int not in random_ls:
                break
        random_ls.append(random_int)
    return random_ls
    
##################################################
#TYPE: 1
##################################################

def add_uniform_noise_to_pos(position, height, width, ratio, isX):
    #isX: position is x_coor (1) or y_coor (2)
    #position: actual pos in real image
    #Get interval by ratio (%)
    if height < width:
        min_size = height
    else:
        min_size = width
    
    max_limit = min_size * ratio

    #Add noise
    noise = np.random.uniform(low=-max_limit, high=max_limit)
    #print(noise)
    new_position = position + noise
    if new_position < 0:
        new_position = 0
    else:
        if isX == 1 and new_position > width:
            new_position = width
        elif isX == 2 and new_position > height:
            new_position = height
    
    return int(new_position)

def add_noise_type1_to_grt(x_ls, y_ls, height, width, ratio, max_change_bee):
    x_unit = x_ls
    y_unit = y_ls
    #Convert x, y to 
    x_ls = [int(x_*width) for x_ in x_ls]
    y_ls = [int(y_*height) for y_ in y_ls]
    num_bee = len(x_ls)
    #print("num_bee:" +str(num_bee))

    x_noise = []
    y_noise = []

    while True:
        random_len = random.randint(1, max_change_bee)
        if random_len < num_bee-1:
            print('Thay d?i: '+str(random_len)+' con ong')
            break

    k_ls = gen_random_int_list(num_bee, random_len)

    for i in range(num_bee):
        if i in k_ls:
            random_int = random.randint(0, 100)
            if random_int % 3 == 1:
                noisy = add_uniform_noise_to_pos(x_ls[i], height, width, ratio, 1)   #New pos x
                x_noise.append(norm_val(noisy*1.0/width))
                y_noise.append(y_unit[i])
            elif random_int % 3 == 2:
                noisy = add_uniform_noise_to_pos(y_ls[i], height, width, ratio, 2)   #New pos y
                x_noise.append(x_unit[i])
                y_noise.append(norm_val(noisy*1.0/height))
            else:
                noisy_x = add_uniform_noise_to_pos(x_ls[i], height, width, ratio, 1)   #New pos x
                noisy_y = add_uniform_noise_to_pos(y_ls[i], height, width, ratio, 2)   #New pos y
                x_noise.append(norm_val(noisy_x*1.0/width))
                y_noise.append(norm_val(noisy_y*1.0/height))
        else:
            x_noise.append(x_unit[i])
            y_noise.append(y_unit[i])

    return x_noise, y_noise   #Unit
    
def plot_noise_bee(img_path, txt_path, ratio=0.01, max_change_bee=10):
    img_cvt, coordinates_dt = read_data_bee(img_path, txt_path)
    h, w = img_cvt.shape[0], img_cvt.shape[1]

    x_ls, y_ls = coordinates_dt['x'], coordinates_dt['y']
    x_noise_ls, y_noise_ls = add_noise_type1_to_grt(x_ls, y_ls, h, w, ratio, max_change_bee)

    figure = plt.figure(figsize=(8, 5), dpi=300)
    num_bee = len(x_ls)

    for i in range(num_bee):
        #print(y_ls[i]==y_noise_ls[i])
        e_cor_x = int(x_ls[i]*w)
        e_cor_y = int(y_ls[i]*h)
        e_cor_x_noise = int(x_noise_ls[i]*w)
        e_cor_y_noise = int(y_noise_ls[i]*h)

        cv2.drawMarker(img_cvt, (e_cor_x, e_cor_y), (255, 0, 0), thickness=2)
        # if (e_cor_x_noise!=e_cor_x) or (e_cor_y_noise!=e_cor_y):
        #     cv2.drawMarker(img_cvt, (e_cor_x_noise, e_cor_y_noise), (255, 255, 0), thickness=2)
        cv2.drawMarker(img_cvt, (e_cor_x_noise, e_cor_y_noise), (255, 255, 0), thickness=2)
        
    title = os.path.basename(img_path).replace(".jpg", "") + "_ratio_"+str(ratio)+"_beeMax_"+str(max_change_bee)
    plt.axis('off')
    plt.imshow(img_cvt)
    plt.title(title)
    
def write_type_1_to_text(img_path, txt_path, save_path, ratio=0.01, max_change_bee=10):
    img_cvt, coordinates_dt = read_data_bee(img_path, txt_path)
    h, w = img_cvt.shape[0], img_cvt.shape[1]

    x_ls, y_ls = coordinates_dt['x'], coordinates_dt['y']
    x_noise_ls, y_noise_ls = add_noise_type1_to_grt(x_ls, y_ls, h, w, ratio, max_change_bee)
    num_bee = len(x_ls)
    a_ls = coordinates_dt['a']
    b_ls = coordinates_dt['b']
    namefile = os.path.basename(txt_path)
    file_dir = os.path.join(save_path, namefile)
    #print(file_dir)
    with open(file_dir, 'w') as f:
        for i in range(num_bee):
            line_txt = '0 ' + str(x_noise_ls[i]) + ' ' + str(y_noise_ls[i]) + ' ' + str(a_ls[i]) + ' ' + str(b_ls[i]) + '\n'
            f.write(line_txt)
    f.close()

def write_type_1_train(old_folder, new_folder, ratio=0.01, max_change_bee=10):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for textfile in os.listdir(old_folder):
        print("Process: " + textfile)
        if textfile.endswith('.txt'):
            text_dir = os.path.join(old_folder, textfile)
            img_dir = os.path.join(old_folder, textfile.replace('.txt','.jpg'))
            #write_type_1_to_text(img_dir, text_dir, 100, 0.1, new_folder)
            write_type_1_to_text(img_dir, text_dir, new_folder, ratio, max_change_bee)
        print('Save!')
    print('Done')
    print('==============')



##################################################
#TYPE: 2
##################################################

def add_gaussian_noise_to_pos(position, sigma):
    #position: normalized pos in yolo format
    #Get interval by ratio (%)    

    #Add noise
    noise = random.gauss(0, sigma)
    #print(noise)
    new_position = position + noise

    return norm_val(new_position) 

# add_gaussian_noise_to_pos(0.5244, 0.5)

def add_noise_type2_to_grt(x_ls, y_ls, sigma, max_change_bee):
    x_noise = []
    y_noise = []
    num_bee = len(x_ls)
    while True:
        random_len = random.randint(1, max_change_bee)
        if random_len < num_bee-1:
            print('Thay d?i: '+str(random_len)+' con ong')
            break

    k_ls = gen_random_int_list(num_bee, random_len)

    for i in range(num_bee):
        if i in k_ls:
            random_int = random.randint(0, 100)
            if random_int % 3 == 1:
                noisy = add_gaussian_noise_to_pos(x_ls[i], sigma)   #New pos x
                x_noise.append(noisy)
                y_noise.append(y_ls[i])
            elif random_int % 3 == 2:
                noisy = add_gaussian_noise_to_pos(y_ls[i], sigma)   #New pos y
                x_noise.append(x_ls[i])
                y_noise.append(noisy)
            else:
                noisy_x = add_gaussian_noise_to_pos(x_ls[i], sigma)   #New pos x
                noisy_y = add_gaussian_noise_to_pos(y_ls[i], sigma)   #New pos y
                x_noise.append(noisy_x)
                y_noise.append(noisy_y)
        else:
            x_noise.append(x_ls[i])
            y_noise.append(y_ls[i])

    return x_noise, y_noise   #Unit

def plot_noise_bee2(img_path, txt_path, sigma=0.1, max_change_bee=20):
    img_cvt, coordinates_dt = read_data_bee(img_path, txt_path)
    h, w = img_cvt.shape[0], img_cvt.shape[1]

    x_ls, y_ls = coordinates_dt['x'], coordinates_dt['y']
    x_noise_ls, y_noise_ls = add_noise_type2_to_grt(x_ls, y_ls, sigma, max_change_bee)

    figure = plt.figure(figsize=(8, 5), dpi=300)
    num_bee = len(x_ls)

    for i in range(num_bee):
        #print(y_ls[i]==y_noise_ls[i])
        e_cor_x = int(x_ls[i]*w)
        e_cor_y = int(y_ls[i]*h)
        e_cor_x_noise = int(x_noise_ls[i]*w)
        e_cor_y_noise = int(y_noise_ls[i]*h)

        cv2.drawMarker(img_cvt, (e_cor_x, e_cor_y), (255, 0, 0), thickness=2)
        # if (e_cor_x_noise!=e_cor_x) or (e_cor_y_noise!=e_cor_y):
        #     cv2.drawMarker(img_cvt, (e_cor_x_noise, e_cor_y_noise), (255, 255, 0), thickness=2)
        cv2.drawMarker(img_cvt, (e_cor_x_noise, e_cor_y_noise), (255, 255, 0), thickness=2)
        
    title = os.path.basename(img_path).replace(".jpg", "") + "_sigma_"+str(10*sigma)+"_beeMax_"+str(max_change_bee)
    plt.axis('off')
    plt.imshow(img_cvt)
    plt.title(title)

def write_type_2_to_text(img_path, txt_path, save_path, sigma=0.1, max_change_bee=10):
    img_cvt, coordinates_dt = read_data_bee(img_path, txt_path)
    # h, w = img_cvt.shape[0], img_cvt.shape[1]

    x_ls, y_ls = coordinates_dt['x'], coordinates_dt['y']
    x_noise_ls, y_noise_ls = add_noise_type2_to_grt(x_ls, y_ls, sigma, max_change_bee)
    num_bee = len(x_ls)
    a_ls = coordinates_dt['a']
    b_ls = coordinates_dt['b']
    #namefile = os.path.basename(txt_path).replace('.txt', '_type2.txt')
    namefile = os.path.basename(txt_path)
    file_dir = os.path.join(save_path, namefile)
    #print(file_dir)
    with open(file_dir, 'w') as f:
        for i in range(num_bee):
            line_txt = '0 ' + str(x_noise_ls[i]) + ' ' + str(y_noise_ls[i]) + ' ' + str(a_ls[i]) + ' ' + str(b_ls[i]) + '\n'
            f.write(line_txt)
    f.close()

def write_type_2_train(old_folder, new_folder, sigma=8, max_change_bee=20):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for textfile in os.listdir(old_folder):
        print("Process: " + textfile)
        if textfile.endswith('.txt'):
            text_dir = os.path.join(old_folder, textfile)
            img_dir = os.path.join(old_folder, textfile.replace('.txt','.jpg'))
            #write_type_1_to_text(img_dir, text_dir, 100, 0.1, new_folder)
            write_type_2_to_text(img_dir, text_dir, new_folder, sigma, max_change_bee)
        print('Save!')
    print('Done')
    print('==============')

##################################################
#TYPE: 3
##################################################

# def write_type_3_to_text(img_path, txt_path, save_path, alpha=100, ratio=0.01, max_num_change=10):
#     random_int = random.randint(0, 100)
#     if random_int % 4 == 1:
#         print('Type 1: ')
#         write_type_1_to_text(img_path, txt_path, save_path, alpha, ratio)
#     elif random_int % 4 == 2:
#         print('Type 2:')
#         write_type_2_to_text(txt_path, save_path, max_num_change)
#     elif random_int % 4 == 3:
#         print('Type 3:')
#         write_type_2_to_text(txt_path, save_path, max_num_change)
#         new_txt_path = os.path.join(save_path, os.path.basename(txt_path))
#         write_type_1_to_text(img_path, new_txt_path, save_path, alpha, ratio)
#         gen_new_bee(txt_path, save_path, max_num_change)
#     else:
#         print('Change nothing!!!')
        
# def write_type_3_train(folder_path, save_path, alpha=100, ratio=0.01, max_num_change=10):
#     for textfile in os.listdir(folder_path):
#         if textfile.endswith('.txt'):
#             print("Process: " + textfile)
#             text_dir = os.path.join(folder_path, textfile)
#             img_dir = os.path.join(folder_path, textfile.replace('.txt','.jpg'))
#             write_type_3_to_text(img_dir, text_dir, save_path, alpha, ratio, max_num_change)
#             print('Save!')
#     print('Done')
#     print('==============')

def del_random_bee(txt_path, save_path, max_num_del):
    filename = os.path.basename(txt_path)
    save_file = os.path.join(save_path, filename)
    #print(save_file)

    with open(txt_path, "r") as f:
        lines = f.readlines()
    f.close()
    # print('Truoc khi xoa:')
    # print(lines)
    # print('=========')

    numbee = len(lines)
    while True:
        num_del = random.randint(0, max_num_del)
        if num_del < numbee-4:
            print('Delete: '+ str(num_del) + ' bee')
            break

    num_keep = numbee - num_del
    set_bee_keep = set(random.sample(lines, num_keep))
    #print(set_bee_keep)
    newlines = [i for i in lines if i in set_bee_keep]

    # print('Sau khi xoa:')
    # print(newlines)
    # print('=========')

    with open(save_file, 'w') as sf:
        sf.writelines(newlines)
    sf.close()

def gen_new_string(line):
    #print(line)

    line_ele = line.replace('\n','').split(' ')

    x_coor = random.randint(0, 1920)
    x_val = norm_val(x_coor*1.0/1920)
    y_coor = random.randint(0, 1080)
    y_val = norm_val(y_coor*1.0/1080)

    line_ele[1] = str(x_val)
    line_ele[2] = str(y_val)

    newline = ' '.join(line_ele) + '\n'
    #print(newline)

    return newline

def gen_new_bee(txt_path, save_path, max_num_add):
    filename = os.path.basename(txt_path)
    save_file = os.path.join(save_path, filename)
    #print(save_file)

    with open(txt_path, "r") as f:
        lines = f.readlines()
    f.close()
    # print('Truoc khi them:')
    # print(lines)
    # print('=========')

    numbee = len(lines)
    num_add = random.randint(0, max_num_add)
    print('Add: '+ str(num_add) + ' bees')

    for i in range(num_add):
        random_int = random.randint(0, numbee-1)
        random_line = lines[random_int]
        new_line = gen_new_string(random_line)
        lines.append(new_line)


    # print('Sau khi them:')
    # print(lines)
    # print('=========')

    with open(save_file, 'w') as sf:
        sf.writelines(lines)
    sf.close()

def write_type_3_to_text(txt_path, save_path, max_num_change):
    random_int = random.randint(0, 100)
    if not random_int % 2:
        print('Del bees')
        del_random_bee(txt_path, save_path, max_num_change)
    else:
        print('Add bees')
        gen_new_bee(txt_path, save_path, max_num_change)

def write_type_3_train(folder_path, save_path, max_num_change=10):
    for textfile in os.listdir(folder_path):
        if textfile.endswith('.txt'):
            print("Process: " + textfile)
            text_dir = os.path.join(folder_path, textfile)
            write_type_3_to_text(text_dir, save_path, max_num_change)
            print('Save!')
    print('Done')
    print('==============')
    
##################################################
#TYPE: 4: MAIN NOISE
##################################################

def add_uniform_noise_to_pos_4(position, height, width, isX, random_val):
    #Add noise
    noise = np.random.uniform(-random_val, random_val)
    #print(noise)
    new_position = position + noise
    if new_position < 0:
        new_position = position + abs(noise)
    else:
        if isX == 1 and new_position >= width:
            new_position = position - abs(noise)
        elif isX == 2 and new_position >= height:
            new_position = position - abs(noise)

    return int(new_position)

#add_uniform_noise_to_pos(1919, 1080, 1920, 1, 5)

def add_noise_type4_to_grt(x_ls, y_ls, height, width, random_val):
    #Add both x and y and for all the data points
    #Convert x, y to
    x_ls = [int(x_*width) for x_ in x_ls]
    y_ls = [int(y_*height) for y_ in y_ls]
    num_bee = len(x_ls)
    #print("num_bee:" +str(num_bee))

    x_noise = []
    y_noise = []

    for i in range(num_bee):
        noisy_x = add_uniform_noise_to_pos_4(x_ls[i], height, width, 1, random_val)   #New pos x
        noisy_y = add_uniform_noise_to_pos_4(y_ls[i], height, width, 2, random_val)   #New pos y
        x_noise.append(norm_val(noisy_x*1.0/width))
        y_noise.append(norm_val(noisy_y*1.0/height))
        
    return x_noise, y_noise   #Unit
    
def write_type_4_to_text(img_path, txt_path, save_path, random_val):
    img_cvt, coordinates_dt = read_data_bee(img_path, txt_path)
    h, w = img_cvt.shape[0], img_cvt.shape[1]

    for i in range(5):
        x_ls, y_ls = coordinates_dt['x'], coordinates_dt['y']
        x_noise_ls, y_noise_ls = add_noise_type4_to_grt(x_ls, y_ls, h, w, random_val)
        num_bee = len(x_ls)
        a_ls = coordinates_dt['a']
        b_ls = coordinates_dt['b']
        namefile = os.path.basename(txt_path).replace('.txt', '') + '_' + str(i+1) + '.txt'
        file_dir = os.path.join(save_path, namefile)
        #print(file_dir)
        with open(file_dir, 'w') as f:
            for i in range(num_bee):
                line_txt = '0 ' + str(x_noise_ls[i]) + ' ' + str(y_noise_ls[i]) + ' ' + str(a_ls[i]) + ' ' + str(b_ls[i]) + '\n'
                f.write(line_txt)
        f.close()

def write_type_4_train(old_folder, new_folder, random_val):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for textfile in os.listdir(old_folder):
        print("Process: " + textfile)
        if textfile.endswith('.txt'):
            text_dir = os.path.join(old_folder, textfile)
            img_dir = os.path.join(old_folder, textfile.replace('.txt','.jpg'))
            #write_type_1_to_text(img_dir, text_dir, 100, 0.1, new_folder)
            write_type_4_to_text(img_dir, text_dir, new_folder, random_val)
        print('Save!')
    print('Done')
    print('==============')


##################################################
#Main
##################################################

if __name__ == '__main__':
    old_folder = "/home/vuhai/Tung-Bayesian-Bee/Bee_comvis_ver_3_std/Train/"
    new_folder = "/home/vuhai/Tung-Bayesian-Bee/Bee_comvis_ver_noise/type_2/sig_0.05_5/Train/"
    #print('Type 2')
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    # #Type 1
    #write_type_1_train(old_folder=old_folder, new_folder=new_folder, ratio=0.05, max_change_bee=5)
    #Type 2
    #write_type_2_train(old_folder=old_folder, new_folder=new_folder, sigma=0.05, max_change_bee=5)
    # #Type 3
    #write_type_3_train(old_folder, new_folder, 20)

    #Type 4
    write_type_4_train(old_folder, new_folder, 5)    