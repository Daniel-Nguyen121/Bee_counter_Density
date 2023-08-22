import os

def get_file(path):
    for subfolder in sorted(os.listdir(path)):
        print(subfolder)
        sub_dir = os.path.join(path, subfolder)
        ls = os.listdir(sub_dir)
        print('Chua: ' + str(len(ls)))
        print('==========')

if __name__ == '__main__':
    path ="/home/vuhai/Bee_counter/preprocessed_noise_5/"
    
    get_file(path)