import os
from PIL import Image
import numpy as np

def prepare_image(imgf_path,target_size=(32,32)):
    img = Image.open(imgf_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_normalized = img_array/255.0
    return img_normalized

def load_data(data_folder,target_size=(32,32)):
    images=[]
    labels=[]
    parijat_folder=os.path.join(data_folder,"parijat")
    for filename in os.listdir(parijat_folder):
        if filename.endswith(('.png','.jpeg','.jpg')):
            img_path=os.path.join(parijat_folder,filename)
            img_array=prepare_image(img_path,target_size)
            images.append(img_array)
            labels.append(1)
    mango_folder=os.path.join(data_folder,"mango")
    for filename in os.listdir(mango_folder):
        if filename.endswith(('.png','.jpeg','.jpg')):
            img_path=os.path.join(mango_folder,filename)
            img_array=prepare_image(img_path,target_size)
            images.append(img_array)
            labels.append(0)
    money_folder=os.path.join(data_folder,"money-plant")
    for filename in os.listdir(money_folder):
        if filename.endswith(('.png','.jpeg','.jpg')):
            img_path=os.path.join(money_folder,filename)
            img_array=prepare_image(img_path,target_size)
            images.append(img_array)
            labels.append(0)
    return np.array(images), np.array(labels)

if __name__=="__main__":
    images,labels=load_data("data/healthy")

