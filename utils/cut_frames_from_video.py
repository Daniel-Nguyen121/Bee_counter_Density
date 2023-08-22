#To download folder from Google Drive

#gdown https://drive.google.com/drive/folders/1LS4JJ0QLbFZofwkj5_pqjtGLWIUuqznK -O /home/vuhai/Tung-Bayesian-Bee/real_data/image_real --folder --remaining-ok

import os
import cv2
import time
import argparse
from utils.create_dir import create_dir

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='/home/vuhai/Tung-Bayesian-Bee/real_data/video_real',
                        help='video data directory')
    parser.add_argument('--save-dir', default='/home/vuhai/Tung-Bayesian-Bee/real_data/image_real',
                        help='image data directory')
    parser.add_argument('--fps', default=10, help='frame per second')
    args = parser.parse_args()
    return args

def cut_images(video_file, output_dir, frame_rate=6):
    '''
    Cuts images from a video following the specified frame rate.
    '''
    cap = cv2.VideoCapture(video_file)
    #frame
    current_frame = 0

    frame_skip = frame_rate
    prev = 0
    i = 0
    # a variable to keep track of the frame to be saved
    frame_count = 0

    print("Start Processing Video: " + os.path.basename(video_file))
    t1 = time.time()
    time_ls = []
    while cap.isOpened():
        #reading from frame
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            # if video is still left continue creating images
            name = os.path.basename(video_file).replace('.mp4', ' ') + str(frame_count) + '.jpg'
            #name = 'frame' + str(current_frame) + '.jpg'
            output_path = os.path.join(output_dir, name)
            print ('Creating...' + name)
            t = time.time()
            # writing the extracted images
            resize = cv2.resize(frame, (1920, 1080), interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(output_path, resize)
            #cv2.imwrite(output_path, resize)
            i = 0
            tt = time.time()
            time_ls.append(tt-t)
            continue
        i += 1

            # # increasing counter so that it will show how many frames are created
            # current_frame += 1

    # print(current_frame)
    # Release all space and windows once done
    cap.release()
    cv2.destroyAllWindows()
    t2 = time.time()
    mean = sum(time_ls) / len(time_ls)
    print("OK!")
    print("\n=========\n")
    print("Takes: ", str(round(t2-t1, 2)), " s")
    print("Average 1 image takes: ", str(round(mean, 2)))
    
    
def rename_files(folder_path, file_ext, del_str="(1)"):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_ext):
            count += 1
            new_name = file_name.replace(del_str, "")
            print(new_name)
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
    print('Tong: ', str(count))
    

def process_list_video(vid_root_dir, out_root_dir, fps=6):
    print("---------START---------")
    
    #Hive_level
    for hive_folder in os.listdir(vid_root_dir):

        print("=====  Hive: ", hive_folder, "  =====")
        hive_root_dir, hive_save_dir = create_dir(vid_root_dir, out_root_dir, hive_folder)

        for year_folder in os.listdir(hive_root_dir):
            print("==========  Year: ", year_folder, "  ==========")
            year_root_dir, year_save_dir = create_dir(hive_root_dir, hive_save_dir, year_folder)

            for month_folder in os.listdir(year_root_dir):
                print("===============  Month: ", month_folder, "  ===============")
                month_root_dir, month_save_dir = create_dir(year_root_dir, year_save_dir, month_folder)

                for day_folder in os.listdir(month_root_dir):
                    print("====================  Day: ", day_folder, "  ====================")
                    day_root_dir, day_save_dir = create_dir(month_root_dir, month_save_dir, day_folder)

                    for video_filename in os.listdir(day_root_dir):
                        print("=========================  Video: ", video_filename, "  =========================")
                        filename = video_filename.replace(".mp4", "")
                        input_dir = os.path.join(day_root_dir, video_filename)
                        save_dir = os.path.join(day_save_dir, filename)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        
                        cut_images(input_dir, save_dir, fps)
    print(print("---------DONE---------"))
    

if __name__ == "__main__":
    #video_dir = "/home/vuhai/Tung-Bayesian-Bee/real_data/video_real"
    #output_dir = "/home/vuhai/Tung-Bayesian-Bee/real_data/image_real"
    #fps = 10
    #process_list_video(video_dir, output_dir, fps)
    
    args = parse_args()
    fps = int(args.fps)
    process_list_video(args.data_dir, args.save_dir, fps)

