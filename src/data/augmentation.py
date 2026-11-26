import os
CURRENT_PATH = os.getcwd()
ROOT_DIR = CURRENT_PATH.split("PanaromicSegmentation")[0]+"PanaromicSegmentation/"
import sys
sys.path.append(ROOT_DIR)

import torchvision.transforms.functional as TF
import random
import math
import logging
logging.basicConfig(level=logging.INFO)
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import pandas as pd

AUG_IMG_DEST_PATH = "data/augmented/img"
AUG_MASK_DEST_PATH = "data/augmented/mask"


x_trans,y_trans = random.uniform(-0.3,0.3),random.uniform(-0.3,0.3)

def process_aug_row(row,aug_img,aug_mask,suffix,aug_type):
    row_new = row.copy()
    row_new['img_rel_path']= os.path.join(AUG_IMG_DEST_PATH,row['set_name'],get_aug_img_name(row['img_rel_path'],suffix))
    row_new['mask_rel_path']= os.path.join(AUG_MASK_DEST_PATH,row['set_name'],get_aug_mask_name(row['mask_rel_path'],suffix))
    row_new['type']= aug_type
    aug_img.save(os.path.join(ROOT_DIR,row_new['img_rel_path']))
    aug_mask.save(os.path.join(ROOT_DIR,row_new['mask_rel_path']))
    return row_new

def get_file_name_and_extension(file_path):
    # Extract the file name with extension
    file_name = os.path.basename(file_path)
    # Split the file name and extension
    name, extension = os.path.splitext(file_name)
    # Return the file name and extension without the dot
    return name, extension[1:] if extension else ''

def get_aug_img_name(img_rel_path,suffix):
    img_name ,img_ext = get_file_name_and_extension(img_rel_path)
    return img_name+"_"+suffix+"."+img_ext

def get_aug_mask_name(mask_rel_path,suffix):
    mask_name, mask_ext = get_file_name_and_extension(mask_rel_path)
    return mask_name+"_"+suffix+"."+mask_ext

## AUGMENTATION
def rotate_image_label(img,mask,angle):
    img = TF.rotate(img,-angle)
    mask = TF.rotate(mask,-angle)
    return img,mask

def trans_image_label(img,mask,trans):
    x_trans,y_trans=trans[0],trans[1]
    imgw,imgh = img.size
    img  = TF.affine(img,0,translate=[x_trans*imgw,y_trans*imgh],scale = 1,shear = 0)
    mask  = TF.affine(mask,0,translate=[x_trans*imgw,y_trans*imgh],scale = 1,shear = 0)
    return img,mask

def gbcss_image_label(img,mask,prob=0.5):
    count=0
    if random.uniform(0,1)<prob:
        factor = random.choice([round(random.uniform(0.65, 0.9), 1),round(random.uniform(1.1, 1.6), 1)])
        img = TF.adjust_gamma(img,factor)
        count+=1
        
    if random.uniform(0,1)<prob:
        factor = random.choice([round(random.uniform(0.5, 0.9), 1),round(random.uniform(1.1, 1.4), 1)])
        img= TF.adjust_brightness(img, factor)
        count+=1
        
    if random.uniform(0,1)<prob:
        factor = random.choice([round(random.uniform(0.5, 0.9), 1),round(random.uniform(2, 4), 1)])
        img= TF.adjust_contrast(img, factor)
        count+=1
    
    if random.uniform(0,1)<prob:
        factor = random.choice([round(random.uniform(0.1, 0.9), 1),round(random.uniform(2, 4), 1)])
        img= TF.adjust_sharpness(img, factor)
        count+=1
    
    
    return img,mask,count

def all_aug_image_label(img,mask,angle,trans,prob):
    img,mask = rotate_image_label(img,mask,angle)
    img,mask = trans_image_label(img,mask,trans)
    img,mask,_ = gbcss_image_label(img,mask,prob)
    return img,mask
    


def augment_images(row,num_rot=3,num_trans=3,num_gbcss=1,num_all_aug=3,rot_prob=0.8,trans_prob=0.8,gbcss_prob=0.8,all_aug_prob=0.8):
    
    #Initializing augmented dataframe
    aug_df = []
    
    #Getting augmentation destination path
    aug_img_dir_path = os.path.join(ROOT_DIR,AUG_IMG_DEST_PATH,row['set_name'])
    aug_mask_dir_path = os.path.join(ROOT_DIR,AUG_MASK_DEST_PATH,row['set_name'])
    
    #Making destination directory
    if not os.path.exists(aug_img_dir_path):
        os.mkdir(aug_img_dir_path)
    if not os.path.exists(aug_mask_dir_path):
        os.mkdir(aug_mask_dir_path)
    
    #Reading images
    mask = Image.open(os.path.join(ROOT_DIR,row['mask_rel_path']))
    img = Image.open(os.path.join(ROOT_DIR,row['img_rel_path'])).convert('RGB')
    
    #Rotation augmentation
    if random.uniform(0,1)<rot_prob:
        angles = [360-20,360-40,360-60,360-80,360-100,360-120,360-140,360-160,180,20,40,60,80,100,120,140,160]
        samp_angles = random.sample(angles,num_rot)
        
        for angle in samp_angles:
            try:
                aug_img,aug_label = rotate_image_label(img,mask,angle)
                suffix = f"rot_{str(angle)}"
                aug_df.append(process_aug_row(row,aug_img,aug_label,suffix,'rotate'))
            except Exception as e:
                logging.info(f"Failed on {suffix} with img_id={row['img_id']}: "+str(e))
    
    #Translate augmentation
    if random.uniform(0,1)<trans_prob:
        trans_range = (-0.3, 0.3)
        samp_trans = [(random.uniform(*trans_range), random.uniform(*trans_range)) for _ in range(num_trans)]
        
        for trans in samp_trans:
            try:
                aug_img,aug_mask = trans_image_label(img,mask,trans)
                suffix = f"trans_{str(trans[0])}_{str(trans[1])}"
                aug_df.append(process_aug_row(row,aug_img,aug_mask,suffix,'translate'))
                
            except Exception as e:
                logging.info(f"Failed on {suffix} with img_id={row['img_id']}: "+str(e))
    
    #GBCSS augmentation
    if random.uniform(0,1)<gbcss_prob:
        probs=[0.6]*num_gbcss
        
        for prob in probs:
            try:
                aug_img,aug_mask,count = gbcss_image_label(img,mask,prob)
                suffix = f"gbcss_{str(count)}"
                if count!=0:
                    aug_df.append(process_aug_row(row,aug_img,aug_mask,suffix,'gbcss'))
                
            except Exception as e:
                logging.info(f"Failed on {suffix} with img_id={row['img_id']}: "+str(e))
    
    #All augmentation
    if random.uniform(0,1)<all_aug_prob:
        angles = [360-30,360-50,360-70,360-90,360-110,360-130,360-150,360-170,30,50,70,90,110,130,150,170]
        trans_range = (-0.3, 0.3)
        prob=0.16
        
        for i in range(num_all_aug):
            try:
                angle = random.choice(angles)
                trans = (random.uniform(*trans_range), random.uniform(*trans_range))
                aug_img,aug_label = all_aug_image_label(img,mask,angle,trans,prob)
                suffix = f"rot_{str(angle)}_trans_{trans[0]}_{trans[1]}_gbcss"
                aug_df.append(process_aug_row(row,aug_img,aug_label,suffix,'all_aug'))
            except Exception as e:
                logging.info(f"Failed on {suffix} with img_id={row['img_id']}: "+str(e))
    
    return  pd.DataFrame(aug_df).reset_index(drop=True)   
                
            
            
            
    
    