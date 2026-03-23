import os
from PIL import Image
import numpy as np 

print("Starting leaf preparation...")

def prepare_image(img_path,target_size=(32,32)):
	img = Image.open(img_path)
	print(f" Original size: {img.size}")

	img = img.convert('L')

	img = img.resize(target_size)
	print(f" Resized to: {img.size}")

	img_array = np.array(img)
	img_array = img_array/255.0
	return img_array

def load_data(data_folder,target_size=(32,32)):
	images=[]
	labels=[]

	parijat_folder = os.path.join(data_folder,'parijat')
	for filename in os.listdir(parijat_folder):
		if filename.endswith(('.jpg','.png','.jpeg')):
			img_path=os.path.join(parijat_folder,filename)
			img_array=prepare_image(img_path,target_size)
			images.append(img_array)
			labels.append(1)
	mango_folder = os.path.join(data_folder,'mango')
	for filename in os.listdir(mango_folder):
		img_path = os.path.join(mango_folder,filename)
		img_array=prepare_image(img_path,target_size)
		images.append(img_array)
		labels.append(0)
	money_folder=os.path.join(data_folder,'money-plant')
	for filename in os.listdir(money_folder):
		img_path=os.path.join(money_folder,filename)
		img_array=prepare_image(img_path,target_size)
		images.append(img_array)
		labels.append(0)

	return np.array(images),np.array(labels)

images,labels = load_data('data/healthy')

print(f"Loaded {len(images)} images")
print(f"Images shape: {images.shape}")
print(f"labels shpae: {labels.shape}")

flattened_images = images.reshape(images.shape[0],-1)
print(f"flattened shape: {flattened_images.shape}")

