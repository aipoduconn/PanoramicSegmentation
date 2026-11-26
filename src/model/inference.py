
import ast
import os
CURRENT_PATH = os.getcwd()
ROOT_DIR = CURRENT_PATH.split("PanaromicSegmentation")[0]+"PanaromicSegmentation/"
import sys
sys.path.append(ROOT_DIR)
import src.data.utils as ut
from src.data.utils import IDX2CAT
from src.model.train import PanImgDataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms
import pandas as pd
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights,deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
from torchmetrics import  Accuracy, F1Score
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm
import random
import src.model.train as tr
from src.model import models
import mlflow
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
import mlflow
import tempfile
import logging

class PanInference:
    def __init__(self,model_name,state_dict_uri,root_dir,num_classes=33,img_size=(256,256)):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.model = models.PanSegModel(model_name,self.num_classes)
        self.model.load_state_dict(mlflow.pytorch.load_state_dict(state_dict_uri))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_mask = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(img_size)
            
        ])
        self.pred_tensor = None
        self.true_tensor = None
        self.iou = MeanIoU(num_classes=self.num_classes,per_class=True)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes,average=None)
        self.f1 = F1Score(task="multiclass", num_classes=self.num_classes,average=None)
        self.metrics_df = None
        self.metrics=None
        self.img_size=img_size
    
    def get_pred_true_img_mask(self,img_rel_path):
        img_path = os.path.join(self.root_dir,img_rel_path)
        img = Image.open(img_path).convert('RGB')
        trans_img = self.transform_img(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(trans_img.to(self.device))
        pred_mask = out['out'].argmax(1).detach().cpu()
        pred_mask = transforms.functional.resize(pred_mask,(img.height,img.width))
        
        mask_path = img_path.replace("raw_imgs","mask")+".png"
        true_mask = Image.open(mask_path)
        true_mask_tensor = transforms.functional.pil_to_tensor(true_mask)
        img_tensor = transforms.functional.pil_to_tensor(img)
        return ut.get_masked_img(img_tensor.clone(),pred_mask),ut.get_masked_img(img_tensor.clone(),true_mask_tensor)
    
    def get_pred_true_mask_tensor(self,img_rel_path):
        img_path = os.path.join(self.root_dir,img_rel_path)
        img = Image.open(img_path).convert('RGB')
        trans_img = self.transform_img(img).unsqueeze(0)
        with torch.no_grad():
            out = self.model(trans_img.to(self.device))
        pred_mask = out['out'].argmax(1).detach().cpu()
        pred_mask = transforms.functional.resize(pred_mask,(img.height,img.width))
        
        mask_path = img_path.replace("raw_imgs","mask")+".png"
        true_mask = Image.open(mask_path)
        true_mask_tensor = transforms.functional.pil_to_tensor(true_mask)
        
        return pred_mask,true_mask_tensor
    
    def calc_predictions(self,csv_path,batch_size=16,test_sets=None):
        if test_sets:
            data = pd.read_csv(csv_path).query("set_name in @test_sets")
        else:
            data = pd.read_csv(csv_path)
        val_dataset = PanImgDataset(data,self.root_dir,self.img_size[0],self.img_size[1])
        dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for batch in tqdm(dataloader):
            imgs = batch['img']
            masks = batch['mask']

            with torch.no_grad():
                pred = self.model(imgs.to(self.device))['out']
            pred_masks = pred.argmax(1).detach().cpu()
            iou = self.iou(pred_masks,masks)
            self.accuracy(pred_masks,masks)
            self.f1(pred_masks,masks)
            
        self.metrics_df = pd.DataFrame({
            "iou":self.iou.compute(),
            "accuracy":self.accuracy.compute(),
            "f1": self.f1.compute()
        },index=[IDX2CAT[i] for i in range(self.num_classes)])
        
        self.metrics = {
        "test_iou_bg":self.metrics_df["iou"].mean(),
        "test_accuracy_bg":self.metrics_df["accuracy"].mean(),
        "test_f1_bg":self.metrics_df["f1"].mean(),
        "test_iou_no_bg":self.metrics_df.drop("BACKGROUND")["iou"].mean(),
        "test_accuracy_no_bg":self.metrics_df.drop("BACKGROUND")["accuracy"].mean(),
        "test_f1_no_bg":self.metrics_df["f1"].drop("BACKGROUND").mean(),
    }
   
    def plot_scores(self, scores, categories, title_name='IoU Scores by Category', y_name='IoU Score', x_name='Categories'):
        # Check if the lengths of categories and scores match
        if len(categories) != len(scores):
            raise ValueError("The length of categories and scores must be the same.")

        # Calculate the average score
        average_score = sum(scores) / len(scores)

        # Create the bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(categories, scores, color='skyblue')

        # Add a horizontal dashed line at the average score
        plt.axhline(y=average_score, color='red', linestyle='--', linewidth=1, label=f'Average Score: {average_score:.2f}')

        # Add titles and labels
        plt.title(title_name)
        plt.xlabel(x_name)
        plt.ylabel(y_name)

        # Display the exact score on each bar
        for i, score in enumerate(scores):
            plt.text(i, score + 0.01, f'{score:.2f}', ha='center', va='bottom')

        # Set y-axis limit (optional)
        plt.ylim(0, 1)

        # Add a legend
        plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a BytesIO object
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)  # Move to the beginning of the buffer
        plt.close()  # Close the plot to free memory

        # Convert BytesIO buffer to PIL Image
        return Image.open(img_buffer)
    
    def get_iou_bar_plot(self):
        no_bg_df = self.metrics_df.drop("BACKGROUND")
        return self.plot_scores(no_bg_df["iou"].tolist(),no_bg_df.index.tolist())



def resize_image_proportionately(image, scale_factor):
    """
    Resizes a PIL image proportionately by a given scale factor.

    Parameters:
    - image (PIL.Image.Image): The PIL image to resize.
    - scale_factor (float): The factor by which to scale the image. E.g., 0.5 for 50% of original size.

    Returns:
    - resized_image (PIL.Image.Image): The proportionally resized PIL image.
    """
    # Calculate the new size
    original_size = image.size
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    
    # Resize the image
    resized_image = image.resize(new_size, Image.LANCZOS)
    
    return resized_image

def mlflow_log_df(df, fname):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        temp_file_path = fname
        
        # Save the DataFrame to the temporary file as CSV
        df.to_csv(temp_file_path, index=False)
        
        # Log the CSV file as an artifact
        mlflow.log_artifact(temp_file_path, artifact_path='csv')
    
    # Remove the temporary file
    os.remove(temp_file_path)


def mlflow_log_img(pil_img,fname):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_file_path = fname
        
        # Save the PIL image to the temporary file
        pil_img.save(fname, format='PNG')
    
        # Log the image file as an artifact
        mlflow.log_artifact(temp_file_path, artifact_path='imgs')
    
    # Remove the temporary file
    os.remove(temp_file_path)


  

def infer_and_log(state_dict_path):
    batch_size = 16
    test_sets =["pan_set_1","pan_set_2","pan_set_3","pan_set_4","pan_set_5","pan_set_6","pan_set_7"]
    img_paths = [
        "data/raw_imgs/pan_set_3/img_32.jpg",
        "data/raw_imgs/pan_set_3/img_54.jpg",
        "data/raw_imgs/pan_set_3/img_78.jpg"  
    ]
    
    tracking_server_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_server_uri)
    run_id = state_dict_path.split("/")[5]
    run = mlflow.get_run(run_id)
    params = run.data.params
    
    model_name = ast.literal_eval(params.get("model")).get("name")
    num_classes=ast.literal_eval(params.get("model")).get("num_seg_class")
    img_size = ast.literal_eval(params.get("img_size",str((512,512))))
    
    pi = PanInference(model_name,state_dict_path,ROOT_DIR,num_classes,img_size=img_size)

    csv_path = "/home/ubuntu/Documents/PanaromicSegmentation/data/metadata/test_data.csv"
    pi.calc_predictions(csv_path,batch_size,test_sets)
    
    # print("Logging starts..")
    # mlflow.start_run(run_id=run_id)
    
    # # Logging starts
    # mlflow.log_metrics(pi.metrics)

    # mlflow_log_df(pi.metrics_df,"metrics_df.csv")

    # mlflow_log_img(pi.get_iou_bar_plot(),"iou_plot.png")

    
    # for img_path in img_paths:
    #     pred_img,_ = pi.get_pred_true_img_mask(img_path)
    #     pred_img =resize_image_proportionately(pred_img,1/4)
    #     img_name = "_".join(img_path.split("/")[-2:]).split(".")[0]+".png"
    #     mlflow_log_img(pred_img,img_name)
        
    # mlflow.end_run()
    
    print("logging finished!!!")
    print(pi.metrics)
        
def main():
    state_dict_paths =[
        # "s3://<YOUR_BUCKET>/artifacts/148418839595485829/166130c441cf417a9ef389c4b1f5ca19/artifacts/models/model_epoch_48",
        # "s3://<YOUR_BUCKET>/artifacts/148418839595485829/de3758fca14b4912beda3ba4d85ac716/artifacts/models/model_epoch_52",
        # "s3://<YOUR_BUCKET>/artifacts/148418839595485829/515bf4e6bd7440b4a1db1909c83cd73a/artifacts/models/model_epoch_66",
        # "s3://<YOUR_BUCKET>/artifacts/148418839595485829/748a345b99234947b7128792bd45115b/artifacts/models/model_epoch_86",
        # "s3://<YOUR_BUCKET>/artifacts/148418839595485829/bfe729d927194e2385fda3960f273606/artifacts/models/model_epoch_80",
        "s3://<YOUR_BUCKET>/artifacts/148418839595485829/69ea416e875d433ea5326bf38832cf6b/artifacts/models/model_epoch_74",#512x512 peaceful-fox
        # "s3://<YOUR_BUCKET>/artifacts/148418839595485829/ea363c06f8054d588a1bcb13367742fb/artifacts/models/model_epoch_44",
        # "s3://<YOUR_BUCKET>/artifacts/148418839595485829/fe55f030212e4763ac35463ab03c77bc/artifacts/models/model_epoch_44"
        ]
    for sd_path in state_dict_paths:
        infer_and_log(sd_path)


if __name__ == "__main__":
    main()


        




