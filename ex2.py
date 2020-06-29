import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.color import rgb2lab

PATH = "/Users/dauren/Desktop/"
VIDEO_NAME = "dog_video.mp4"
OUTPUT_VIDEO_NAME = "dog_video_output.mp4"
MODEL_NAME = "autoencoder700.model"


def show_video():
	cap = cv2.VideoCapture(PATH + VIDEO_NAME)
	cap2 = cv2.VideoCapture(PATH + OUTPUT_VIDEO_NAME)

	dim = (640, 320)

	while True:
		ret, frame = cap.read()
		ret2, frame2 = cap2.read()
		if ret:
			colorFrame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
			grayFrame = cv2.cvtColor(colorFrame, cv2.COLOR_RGB2GRAY)
			trainedFrame = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)
			
			cv2.imshow('Original', colorFrame)
			cv2.imshow('Grayscale', grayFrame)
			cv2.imshow("Trained", trainedFrame)
		
		else:
			print('no video')
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

		if (cv2.waitKey(1) == 27):
		    break

	cap.release()
	cv2.destroyAllWindows()

def predict(input_images):
	model = tf.keras.models.load_model(PATH + MODEL_NAME)
	result = model.predict(input_images)
	result *= 128
	return result

def video_to_images(video_path):
	images = []
	dim = (256, 256)
	cnt = 1
	
	cap = cv2.VideoCapture(video_path)
	success, frame = cap.read()
	
	while success:
		images.append(cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) * 1. / 255)
		success, frame = cap.read()

		print('Saved image #', cnt)
		cnt = cnt + 1

	return images

def images_to_video(images):
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	height, width = (256, 256)  
	video = cv2.VideoWriter(PATH + OUTPUT_VIDEO_NAME, fourcc, 20.0, (height, width))  

	# Appending the images to the video one by one 
	for image in images:
		image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
		# cv2_imshow(image)
		if not video.write(image_bgr):
		  print('error')

	# Deallocating memories taken for window creation 
	cv2.destroyAllWindows()  
	video.release()  # releasing the video generated

def lab_to_rgb(input, output):
	rgb_images = []
	for x, y in zip(input, output):
		out = np.zeros((256, 256, 3))
		out[:,:,0] = x[:,:,0]
		out[:,:,1:] = y
		rgb_images.append(lab2rgb(out))

	return rgb_images

def generate_video():
	video_images = video_to_images(PATH + VIDEO_NAME)
	input_images = [(rgb2lab(img)[:,:,0]) for img in video_images]
	target_images = [(rgb2lab(img)[:,:,1:]) for img in video_images]
	result = predict(input_images)
	
	images = lab2rgb(input_images, result)
	images_to_video(images)
	


if __name__ == '__main__':
	generate_video()
	show_video()