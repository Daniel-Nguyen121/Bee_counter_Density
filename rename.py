import os

def rename_files(folder_path, file_ext):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_ext):
            count += 1
            new_name = file_name.replace("(\'", "").replace("\',)", "")
            print(new_name)
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
    print('Tong: ', str(count))

def rename_out_files(folder_path, file_ext):
    count = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_ext):
            count += 1
            new_name = file_name.replace("_", " ")
            print(new_name)
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))
    print('Tong: ', str(count))

if __name__ == "__main__":
    
    root_dir = "/home/vuhai/Tung-yolov5-Bee/runs/detect"
    for i in range(52, 65):
      exp = "exp" + str(i)
      exp_dir = os.path.join(root_dir, exp)
      for labels in os.listdir(exp_dir):
          labels_dir = os.path.join(exp_dir, labels)
          rename_out_files(labels_dir, ".txt")