import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    # Move the files rather than copy because of space limitations in the workspace.
    
    train_val_dir=data_dir+"/train_validation/"
    file_paths=glob.glob(os.path.join(train_val_dir,"*.tfrecord"))
  
    # randomize list to randomize file copying
    random.shuffle(file_paths)
 
    train_dir = data_dir+"/train/"
    test_dir = data_dir+"/test/"
    val_dir = data_dir+"/val/"
    
    # 80% Training
    beg=0
    end=beg + int(0.8*len( file_paths))
    for file in file_paths[beg:end]:
        dest_path=os.path.join(train_dir,os.path.basename(file))
        shutil.move(file,dest_path)
    print("Training Done \n")
   
    # 10% Evaluation
    beg=end
    end=beg + int(0.1*len(file_paths))
    for file in file_paths[beg:end]:
        dest_path=os.path.join(val_dir,os.path.basename(file))
        shutil.move(file,dest_path)
    print("Validation Done \n")    
        
    # 10% testing
    beg = end
    for file in file_paths[beg:]:
        dest_path=os.path.join(test_dir,os.path.basename(file))
        shutil.move(file,dest_path)
    print("Testing Done \n")
    
    return

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)