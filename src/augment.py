"""
Data augmentaion module for leaf images.
Creates variations of images to improve generalization.
"""

import numpy as np 
import random
from scipy.ndimage import rotate, shift, zoom
from PIL import Image

def random_roation(image: np.ndarray, max_angle:float=1q5) -> np.ndarray:
	"""
	Rotate image by random angle between -max_angle and +max_angel.

	Args:
		image: Input image array (H,W,3)
		max_angle: Maximum roation angel in degrees

	Returns:
		Rotated image
	"""

	angle  = random.uniform(-max_angle,+max_angle)
	roatated = rotate(image,angle,reshape=False, order=1)
	return np.clip(roatated,0,1)

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
		img = random_roation(img, p["angle"])
		



