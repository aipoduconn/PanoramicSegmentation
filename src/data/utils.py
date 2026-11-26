import os
CURRENT_PATH = os.getcwd()
ROOT_DIR = CURRENT_PATH.split("PanaromicSegmentation")[0]+"PanaromicSegmentation/"
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import pandas as pd
import shutil
from tqdm import tqdm
import os
import paramiko
from scp import SCPClient




CAT2IDX = {'BACKGROUND':0,
    'UR8': 1,
 'UR7': 2,
 'UR6': 3,
 'UR5': 4,
 'UR4': 5,
 'UR3': 6,
 'UR2': 7,
 'UR1': 8,
 'UL1': 9,
 'UL2': 10,
 'UL3': 11,
 'UL4': 12,
 'UL5': 13,
 'UL6': 14,
 'UL7': 15,
 'UL8': 16,
 'LL8': 17,
 'LL7': 18,
 'LL6': 19,
 'LL5': 20,
 'LL4': 21,
 'LL3': 22,
 'LL2': 23,
 'LL1': 24,
 'LR1': 25,
 'LR2': 26,
 'LR3': 27,
 'LR4': 28,
 'LR5': 29,
 'LR6': 30,
 'LR7': 31,
 'LR8': 32}

CAT2COLOR = {'UR8': '#c71cc9',
 'UR7': '#c19a20',
 'UR6': '#130ba7',
 'UR5': '#d1e280',
 'UR4': '#9a5be1',
 'UR3': '#94ed56',
 'UR2': '#986ced',
 'UR1': '#f15abe',
 'UL1': '#59fefb',
 'UL2': '#965124',
 'UL3': '#ae1ae5',
 'UL4': '#3418ab',
 'UL5': '#c7df06',
 'UL6': '#e26b73',
 'UL7': '#3ffaf6',
 'UL8': '#fa2e44',
 'LL8': '#4cad17',
 'LL7': '#f5d219',
 'LL6': '#df08cd',
 'LL5': '#f71949',
 'LL4': '#00c7fc',
 'LL3': '#1f14e0',
 'LL2': '#e57e8d',
 'LL1': '#f93de6',
 'LR1': '#7505db',
 'LR2': '#f5ec00',
 'LR3': '#bc4224',
 'LR4': '#000000',
 'LR5': '#730bbb',
 'LR6': '#98e976',
 'LR7': '#a90e38',
 'LR8': '#0f0df4'}

ID2IDX = {36: 1,
 37: 2,
 40: 3,
 41: 4,
 49: 5,
 50: 6,
 51: 7,
 52: 8,
 53: 9,
 54: 10,
 55: 11,
 56: 12,
 57: 13,
 58: 14,
 59: 15,
 60: 16,
 61: 17,
 62: 18,
 63: 19,
 64: 20,
 65: 21,
 66: 22,
 67: 23,
 68: 24,
 69: 25,
 70: 26,
 71: 27,
 72: 28,
 73: 29,
 74: 30,
 75: 31,
 76: 32}

IDX2CAT = {v:k for k,v in CAT2IDX.items()}



def download_folder(
    remote_folder, local_folder, remote_host, remote_user, remote_password=None, port=22
):
    if not local_folder.endswith("/"):
        local_folder += "/"
    
    os.makedirs(local_folder, exist_ok=True)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_host, username=remote_user, password=remote_password, port=port)
    scp = SCPClient(ssh.get_transport())

    stdin, stdout, stderr = ssh.exec_command(f"find {remote_folder} -type f")
    remote_files = [line.strip() for line in stdout.readlines()]

    with tqdm(total=len(remote_files), desc="Downloading files") as pbar:
        for remote_path in remote_files:
            relative_path = os.path.relpath(remote_path, remote_folder)
            local_path = os.path.join(local_folder, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            scp.get(remote_path, local_path)
            pbar.update(1)

    scp.close()
    ssh.close()


def generate_masks(coco, image_id, category_ids):
    # Load image
    image_info = coco.loadImgs(image_id)[0]
    
    # Initialize mask
    mask = Image.new('L', (image_info['width'],image_info['height']), 0)
    draw = ImageDraw.Draw(mask)

    # Iterate through annotations and draw masks for specified categories
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=category_ids, iscrowd=None))
    for ann in annotations:
        segmentation = ann['segmentation']
        for seg in segmentation:
            draw.polygon(seg, fill=ID2IDX[ann['category_id']])

    return mask


def hex_to_rgb(hex_code):
    # Remove the '#' if present
    hex_code = hex_code.lstrip('#')
    
    # Convert the hex code to RGB
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    
    return list(rgb)


def get_processed_df(json_file_name,manual_download_dir="data/annots/manual_download/",mask_output_dir="data/mask/",raw_img_dir="data/raw_imgs/"):
    if json_file_name.split(".")[-1]!='json':
        raise("Only json file accepted")
    set_name = json_file_name.split(".")[0]
    coco_annotation_path = os.path.join(ROOT_DIR,manual_download_dir,f"{set_name}.json")
    output_dir = os.path.join(ROOT_DIR,mask_output_dir,set_name)
    
    coco = COCO(coco_annotation_path)

    # Specify categories for which you want to generate segmentation masks
    category_ids = coco.getCatIds(catNms=CAT2IDX.keys()) 

    os.makedirs(output_dir, exist_ok=True)

    # Iterate through images and generate masks
    image_ids = coco.getImgIds()
    new_meta = []
    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]
        if image_info['annotated']:
            try:
                mask = generate_masks(coco, image_id, category_ids)
                # Save mask
                mask_filename = f"{image_info['file_name']}.png"
                mask_path = os.path.join(output_dir, mask_filename)
                mask.save(mask_path)
                new_meta.append({
                    'img_id':image_id,
                    'img_rel_path':f"{raw_img_dir}{set_name}/{image_info['file_name']}",
                    'mask_rel_path':f"{mask_output_dir}{set_name}/{mask_filename}",
                    "set_name": set_name,
                    "type":"original"
                })
            except Exception as e:
                print("Mask not processed",str(e))
            
    return pd.DataFrame(new_meta)

def get_masked_img(img_tensor,mask_tensor,alpha=0.5):
    img_tensor_copy = img_tensor.clone()
    for idx in range(1,33):
        rgb = hex_to_rgb(CAT2COLOR[IDX2CAT[idx]])
        color = torch.tensor(rgb,dtype=torch.uint8).reshape(3,1)
        img_tensor[:,mask_tensor[0]==idx]=color

    return Image.blend(transforms.functional.to_pil_image(img_tensor_copy),transforms.functional.to_pil_image(img_tensor),alpha=alpha) 

def get_pil_masked_img(pil_img,pil_mask,alpha=0.5):
    mask_tensor = transforms.functional.pil_to_tensor(pil_mask)
    img_tensor = transforms.functional.pil_to_tensor(pil_img)
    return get_masked_img(img_tensor,mask_tensor,alpha)

def get_url_masked_img(img_url,mask_url,alpha=0.5):
    mask = Image.open(mask_url)
    img = Image.open(img_url).convert('RGB')
    return get_pil_masked_img(img,mask,alpha)

def get_row_masked_img(row,alpha=0.5):
    img_path = os.path.join(ROOT_DIR,"data",row['img_rel_path'])
    mask_path = os.path.join(ROOT_DIR,"data",row['mask_rel_path'])
    return get_url_masked_img(img_path,mask_path,alpha)
    
def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting directory '{directory_path}': {e}")

def delete_files_in_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Check if it's a file
        if os.path.isfile(file_path):
            try:
                # Delete the file
                if not filename.startswith("."):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def delete_directories_in_directory(dir_path):
    dir_names = os.listdir(dir_path)
    for dn in dir_names:
        try:
            if dn!=".gitkeep":
                delete_directory(os.path.join(dir_path,dn))
        except Exception as e:
            print(str(e))
    