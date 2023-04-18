from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

def network_test_no_args(model, image):
	# construct the argument parser and parse the arguments
	#ap = argparse.ArgumentParser()
	#ap.add_argument("-m", "--model", required=True,
		#help="path to trained model model")
	#ap.add_argument("-i", "--image", required=True,
		#help="path to input image")
	#args = vars(ap.parse_args())
	# load the image
	image = cv2.imread(image)
	orig = image.copy()
	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# load the trained convolutional neural network
	print("[INFO] loading network...")
	model = load_model(model)
	# classify the input image
	(notHex, hex) = model.predict(image)[0]

	# build the label
	label = "Hex Maniac" if hex > notHex else "Not Hex Maniac"
	proba = hex if hex > notHex else notHex
	label = "{}: {:.2f}%".format(label, proba * 100)
	# draw the label on the image
	output = imutils.resize(orig, width=400)
	outputtext = "{}".format(label)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)
	# show the output image
	cv2.imshow("Output", output)
	cv2.waitKey(0)
	return outputtext


def network_test_with_args():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", required=True,
		help="path to trained model model")
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())
	# load the image
	image = cv2.imread(args["image"])
	orig = image.copy()
	# pre-process the image for classification
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# load the trained convolutional neural network
	print("[INFO] loading network...")
	model = load_model(args["model"])
	# classify the input image
	(notHex, hex) = model.predict(image)[0]

	# build the label
	label = "Hex Maniac" if hex > notHex else "Not Hex Maniac"
	proba = hex if hex > notHex else notHex
	label = "{}: {:.2f}%".format(label, proba * 100)
	# draw the label on the image
	output = imutils.resize(orig, width=400)
	cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 255, 0), 2)
	# show the output image
	cv2.imshow("Output", output)
	cv2.waitKey(0)

if __name__ == '__main__':
    print(network_test_no_args("C:\\Users\\Stuff\\Downloads\\Schoolwork\\Personal Projects\\Untitled Tensorflow Project\\IsThisHexManiac\\hexsite\\backend\\hex_maniac.model", 
			       "C:\\Users\\Stuff\\Downloads\\Schoolwork\\Personal Projects\\Untitled Tensorflow Project\\IsThisHexManiac\\hexsite\\backend\\007.png"))