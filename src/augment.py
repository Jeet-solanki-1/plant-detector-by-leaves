"""
Data augmentaion module for leaf images.
Creates variations of images to improve generalization.
"""

import numpy as np 
import random
from scipy.ndimage import rotate, shift, zoom
from PIL import Image

def random_rotation(image: np.ndarray, max_angle:float = 15) -> np.ndarray:
	"""
	Rotate image by random angle between -max_angle and +max_angel.

	Args:
		image: Input image array (H,W,3)
		max_angle: Maximum roation angel in degrees

	Returns:
		Rotated image
	"""

	angle  = random.uniform(-max_angle,+max_angle)
	rotated = rotate(image,angle,reshape=False, order=1)
	return np.clip(rotated,0,1)

def random_flip(image:np.ndarray) -> np.ndarray:
	"""
	Randomly flip image horizontally.
	"""
	if random.random() > 0.5:
		return np.fliplr(image)
	return image

def random_brightness(image: np.ndarray, max_delta=0.2) -> np.ndarray:
	"""
	Adjust brightness randomly.

	Args:
		image: Input image array (0-1 range)
		max_delta: Maximum brightness change (0.2 = +-20%)
	"""
	factor = 1+ random.uniform(-max_delta, max_delta)
	adjusted = image * factor
	return np.clip(adjusted,0,1)

def random_shift(image: np.ndarray, max_shift: float=0.1) -> np.ndarray:
	"""
	Shift image horizontally and vertically.

	Args:
		image: Input image array (H,W,3)
		max_shift: Maximum shift as fraction of image size
	"""
	h,w = image.shape[:2]
	shift_h=int(h*random.uniform(-max_shift, max_shift))
	shift_w=int(w*random.uniform(-max_shift, max_shift)) 
	shifted = shift(image, [shift_h, shift_w, 0], mode='nearest')
	return shifted

def random_zoom(image: np.ndarray, max_zoom: float=0.1) -> np.ndarray:
	"""
	Zoom in or out slightly.

	Args:
		image: Input image array (H,W,3)
		max_zoom: Maximum zoom factor (0.1 = +-10%)
	"""
	h,w = image.shape[:2]
	zoom_factor = 1+ random.uniform(-max_zoom, max_zoom)
	zoomed = zoom(image, [zoom_factor,zoom_factor,1],order=1)

	#Crop or pad to original size
	if zoomed.shape[0]>h:
		start = (zoomed.shape[0]-h)//2
		zoomed = zoomed[start:start+h,start:start+w]
	else:
		pad_h = (h - zoomed.shape[0])//2
		pad_w = (w - zoomed.shape[1])//2
		zoomed = np.pad(zoomed, ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='edge')
	return np.clip(zoomed[:h,:w],0,1)

def augment_image(image: np.ndarray, intensity:str = "medium") -> np.ndarray:
	"""
	Apply a random cmbination of augmentations.

	Args:
		image: Input image array (H,W,3)
		intensity: "light", "medium", or "heavy"

	Returns:
		Augmented image
	"""

	params = {
		"light": {"angle": 5,"brightness": 0.1,"shift": 0.05, "zoom": 0.05},
        "medium": {"angle": 15, "brightness": 0.2, "shift": 0.1, "zoom": 0.1},
        "heavy": {"angle": 30, "brightness": 0.3, "shift": 0.15, "zoom": 0.15}
	}

	p = params[intensity]

	#Apply random transformations (not all, random subset)
	img = image.copy()

	#Rotatio (always apply for medium/heavy)
	if random.random()>0.3:
		img = random_rotation(img, p["angle"])

	#Flip (50% chance)
	img = random_flip(img)

	#Brightness
	if random.random()>0.3:
		img = random_brightness(img, p["brightness"])

	#Shift 
	if random.random()>0.5:
		img = random_shift(img, p["shift"])

	#Zoom
	if random.random() > 0.5:
		img = random_zoom(img,p["zoom"])


	return img
