import pandas as pd
import numpy as np
import os
import scipy
from joblib import Parallel, delayed
import pathlib
import matplotlib.pyplot as plt
import cv2
import subprocess


def cleanup_python_processes():
    # Run the first command
    result = subprocess.run(['aef'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

    # Filter the result using grep to find the line containing "python"
    grep_result = subprocess.run(['grep', 'python'], input=result.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

    # Filter the result again using grep to exclude the line containing "labhub"
    grep_result = subprocess.run(['grep', '-v', 'labhub'], input=grep_result.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

    # Use awk to print the second column of the result
    awk_result = subprocess.run(['awk', '-v', 'x=2', '{print $x}'], input=grep_result.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')

    # Split the result into separate lines
    lines = awk_result.stdout.strip().split('\n')

    # Loop through each line and kill the process using kill -9
    for line in lines:
        subprocess.run(['kill', '-9', line])



def crop_images(row):
    
    # read_images
    dcm_path,laterality,export_path = row['path'], row['laterality'], row['dump_path']
    data = pydicom.read_file(dcm_path)
    img = data.pixel_array
    img = (img - img.min()) /(img.max() - img.min())
    if data.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img
    img *= 255
    img = np.uint8(img)
    
    crop_df = pd.DataFrame(img)
    unique_val = crop_df.nunique().reset_index().rename(columns={0:'nunique'})
    laterality_ = -1 if laterality == "L" else 1
    crop_x = unique_val[unique_val['nunique']>2].iloc[laterality_]['index']
    if laterality == "L" :
        crop_image = img[:,:crop_x+20]
    else:
        crop_image = img[:, crop_x-20:]
        
    export_dir = export_path.split(str(row['patient_id']))[0]+str(row['patient_id'])
    pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
    plt.imsave(export_path, crop_image, cmap='gray')
    

def display_sample_image(df,cols=5,mode="train"):
    df_postive = df[df['cancer']==1].sample(n=10)
    df_neg = df[df['cancer']==0].sample(n=10)
    df = pd.concat([df_postive,df_neg]).reset_index()
    rows = len(df) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))
    ax = ax.ravel()
    for i, row in df.iterrows():
        image = cv2.imread(row['dump_path'])
        true_label = row['cancer']
        predicted_label = "cancer" if true_label == 1 else "non-cancer"
        color = "green" if true_label == 1 else "red"
        ax[i].text( .1,.95, f"{row['image_id']}", ha='left', va='top', fontsize = 15,color = color)
        ax[i].set_title(predicted_label, color=color)
        ax[i].imshow(image)
    plt.tight_layout()
    figure.suptitle(f"Display {mode.capitalize()} sample data")
    plt.show()
    plt.close()
    
    
def get_ml_features(df):
    base_features = [['site_id', 'patient_id', 'image_id', 'laterality', 'view', 'age','implant', 'machine_id']]
    df = df[base_features]
    df['laterality'] = df['laterality'].map({'L' :0 , "R":1})
    df['view'] = df['view'].map({'MLO':0 , "CC":1, "AT":2, "LM": 3, "ML": 4, "LMO": 5})
    df['mean_age'] = df.groupby(['site_id','machine_id'])['age'].transform('mean')
    df.loc[df.age.isnull(),['age']] = df['mean_age']
    return df
        
        
        

