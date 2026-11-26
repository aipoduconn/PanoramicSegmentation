import os
# Repository root (two levels above this file), independent of the working directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
import sys
sys.path.append(ROOT_DIR)
import src.data.utils as ut
from src.data.utils import IDX2CAT
from tqdm import tqdm
import pandas as pd
import src.data.augmentation as aug
from sklearn.model_selection import train_test_split
import os
import argparse
import shutil


MANUAL_ANNOTS_PATH= os.path.join(ROOT_DIR,"data/annots/manual_download")
CODE_MOVED_ANNOTS_PATH=os.path.join(ROOT_DIR,"data/annots/code_moved")
OUTPUT_METADATA_DIR_PATH=os.path.join(ROOT_DIR,"data/metadata")
TEST_SIZE=0.2
AUGMENT=True
NUM_ROT = 3
NUM_TRANS = 2
NUM_GBCSS = 1
NUM_ALL_AUG = 3
ROT_PROB = 1
TRANS_PROB = 1
GBCSS_PROB = 1
ALL_AUG_PROB = 1


def get_aug_df(meta_df,num_rot=3,num_trans=3,num_gbcss=1,num_all_aug=3,rot_prob=1,trans_prob=1,gbcss_prob=1,all_aug_prob=1):
    aug_df = []
    for _,row in tqdm(meta_df.iterrows(),total=len(meta_df)):
        try:
            aug_row_df = aug.augment_images(row,num_rot=num_rot,num_trans=num_trans,num_gbcss=num_gbcss,num_all_aug=num_all_aug,rot_prob=rot_prob,trans_prob=trans_prob,gbcss_prob=gbcss_prob,all_aug_prob=all_aug_prob)
            aug_df.append(aug_row_df)
        except Exception as e:
            print("could not augment due to: ",str(e))
            

    return pd.concat(aug_df)
    

def save_train_test_csv(set_file_name,metadata_path,test_size=0.2,aug=True,
                        num_rot=3,num_trans=3,num_gbcss=1,num_all_aug=3,rot_prob=1,trans_prob=1,gbcss_prob=1,all_aug_prob=1):
    meta_df = ut.get_processed_df(set_file_name)
    train_meta_df, test_meta_df = train_test_split(meta_df,test_size=test_size)
    
    #Augmenting DF
    if aug:
        print("Augmentation starts..")
        aug_df = get_aug_df(train_meta_df,num_rot=num_rot,num_trans=num_trans,num_gbcss=num_gbcss,num_all_aug=num_all_aug,rot_prob=rot_prob,trans_prob=trans_prob,gbcss_prob=gbcss_prob,all_aug_prob=all_aug_prob)
        train_meta_df = pd.concat([train_meta_df,aug_df])
        print("Augmentation finished!")
    
    ##Train metadata saving
    train_fp = os.path.join(metadata_path,"train_data.csv")
    if os.path.isfile(train_fp):
        old_train_metadata = pd.read_csv(train_fp)
        train_meta_df = pd.concat([old_train_metadata,pd.DataFrame(train_meta_df)])
    else:
        train_meta_df=pd.DataFrame(train_meta_df)
    train_meta_df.to_csv(train_fp,index=False)
    
    
    ##Test metadata saving
    test_fp = os.path.join(metadata_path,"test_data.csv")
    if os.path.isfile(test_fp):
        old_test_metadata = pd.read_csv(test_fp)
        test_meta_df = pd.concat([old_test_metadata,pd.DataFrame(test_meta_df)])
    else:
        test_meta_df=pd.DataFrame(test_meta_df)
    test_meta_df.to_csv(test_fp,index=False)
        

def main():
    parser = argparse.ArgumentParser(description="Script for downloading and cleaning raw data")
    
    parser.add_argument('--manual_annots_path',type=str,default=MANUAL_ANNOTS_PATH,help='Folder path of the manually downloaded json annotation file.')
    parser.add_argument('--code_moved_annots_path',type=str,default=CODE_MOVED_ANNOTS_PATH,help='Folder path of the code moved json annotation file.')
    parser.add_argument('--output_metadata_dir_path', type=str,default=OUTPUT_METADATA_DIR_PATH, help='Folder path where the manually downloaded file be stored permanently.')
    parser.add_argument('--test_size', type=float,default=TEST_SIZE, help='Fraction of data to be used as test set.')
    parser.add_argument('--augment', type=bool,default=AUGMENT, help='Whether to perform data augmentation.')
    parser.add_argument('--num_rot',type=int,default=NUM_ROT,help='Number of rotated augmentation.')
    parser.add_argument('--num_trans',type=int,default=NUM_TRANS,help='Number of translated augmentation.')
    parser.add_argument('--num_gbcss',type=int,default=NUM_GBCSS,help='Number of gbcss(brightness etc) augmentation.')
    parser.add_argument('--num_all_aug',type=int,default=NUM_ALL_AUG,help='Number of all combined augmentation.')
    parser.add_argument('--rot_prob',type=float,default=ROT_PROB,help='Fraction of data to be rotation augmented.')
    parser.add_argument('--trans_prob',type=float,default=TRANS_PROB,help='Fraction of data to be translation augmented.')
    parser.add_argument('--gbcss_prob',type=float,default=GBCSS_PROB,help='Fraction of data to be GBCSS augmented.')
    parser.add_argument('--all_aug_prob',type=float,default=ALL_AUG_PROB,help='Fraction of data to be all augmented.')
    args = parser.parse_args()
    
    files = os.listdir(args.manual_annots_path)
    for fn in files:
        if not fn.startswith("."):
            print("Processing",fn)
            save_train_test_csv(fn,
                                args.output_metadata_dir_path,
                                args.test_size,
                                args.augment,
                                args.num_rot,
                                args.num_trans,
                                args.num_gbcss,
                                args.num_all_aug,
                                args.rot_prob,
                                args.trans_prob,
                                args.gbcss_prob,
                                args.all_aug_prob)
            
            #Move the cuurent file from manual folder to actual folder and delete any files from manual folder
            src_path = os.path.join(args.manual_annots_path, fn)
            dst_path = os.path.join(args.code_moved_annots_path, fn)
            shutil.move(src_path, dst_path)
            
            print(fn,"Processed!")
    

if __name__ == "__main__":
    main()