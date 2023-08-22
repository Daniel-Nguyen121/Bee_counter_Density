import os

def create_dir(root_dir, save_dir, folder_name):
    sub_root_dir = os.path.join(root_dir, folder_name)
    sub_save_dir = os.path.join(save_dir, folder_name)
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    
    return sub_root_dir, sub_save_dir