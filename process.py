import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
from PIL import Image
import models

def train():
	# Default Value
	default_width = 28
	default_height = 28
	default_depth = 1
	default_classes = 10
	default_epochs = 5
	default_batch_size = 32
	
	# Load datasets
	dataset = int(input("Choose Dataset:\n[1] MNIST\n[2] SDUMLA\n[3] FV-USM\n[0] Quit\n>> "))
	while True:
		if dataset == 0:
			exit()
		elif dataset == 1:
			(x_train, y_train), (x_test, y_test) = load_dataset("MNIST")
			break
		elif dataset == 2:
			(x_train, y_train), (x_test, y_test) = load_dataset("SDUMLA")	# 106 * 2 * 3 * 6 = 3816 images
			default_width = 320
			default_height = 240
			default_classes = 0
			default_epochs = 1
			default_batch_size = 106
			break
		elif dataset == 3:
			(x_train, y_train), (x_test, y_test) = load_dataset("FV-USM")	# 2 * 123 * 4 * 6 = 5904 images
			default_width = 100
			default_height = 300
			default_classes = 0
			default_epochs = 1
			default_batch_size = 123
			break
		else:
			dataset = int(input("YOU NEED CHOOSE ONE DATASET!\n[1] MNIST\n[2] SDUMLA\n[0] Quit\n>> "))

	# Process data
	x_train = x_train.reshape(-1, default_width, default_height, default_depth)		# expend dimension for 1 channel image
	x_test = x_test.reshape(-1, default_width, default_height, default_depth)		# expend dimension for 1 channel image
	x_train = x_train / 255		# normalize
	x_test = x_test / 255		# normalize
	
	# One hot encoding
	y_train = np_utils.to_categorical(y_train, num_classes = default_classes)
	y_test = np_utils.to_categorical(y_test, num_classes = default_classes)
	
	# Select model
	use = int(input("Choose model:\n[1] LeNet\n[2] AlexNet\n[3] VGG16\n[4] GoogLeNet V1\n[5] ResNet 34\n[0] Quit\n>> "))
	while True:
		if use == 0:
			exit()
		elif use == 1:
			model = models.LeNet(default_width, default_height, default_depth, default_classes)
			break
		elif use == 2:
			model = models.AlexNet(default_width, default_height, default_depth, default_classes)
			break
		elif use == 3:
			model = models.VGG16(default_width, default_height, default_depth, default_classes)
			break
		elif use == 4:
			model = models.GoogLeNet_v1(default_width, default_height, default_depth, default_classes)
			break
		elif use == 5:
			model = models.ResNet34(default_width, default_height, default_depth, default_classes)
			break
		else:	
			use = int(input("YOU NEED CHOOSE ONE MODEL!\n[1] LeNet\n[2] AlexNet\n[3] VGG16\n[4] GoogLeNet V1\n[5] ResNet 34\n[0] Quit\n>> "))
	model.summary()
	model.compile(Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08), loss = 'categorical_crossentropy', metrics = ['accuracy'])

	# Start training
	History = model.fit(x_train, y_train, epochs = default_epochs, batch_size = default_batch_size, validation_data = (x_test, y_test))

	return History


def load_dataset(dataset):
	if dataset == "SDUMLA":
		path = "./Dataset/SDUMLA/"
		testcase = "001"
		hands = ["/left", "/right"]
		fingers = ["/index_", "/middle_", "/ring_"]
		filetype = ".bmp"

		filelist_train = []
		filelist_test = []
		label_train = []
		label_test = []

		for i in range(107):
			for hand in hands:
				for finger in fingers:
					for index in range(1, 7):
						filename = path + testcase + hand + finger + str(index) + filetype
						if index < 5:
							filelist_train.append(filename)
							label_train.append(testcase)
							label_train.append(testcase)
							label_train.append(testcase)
						else:
							filelist_test.append(filename)
							label_test.append(testcase)
							label_test.append(testcase)
							label_test.append(testcase)
			testcase = "0" * (3 - len(str(i + 1))) + str(i + 1)
		
		x_train = np.array([np.array(Image.open(f)) for f in filelist_train])
		y_train = np.array([np.array(l) for l in label_train])
		x_test = np.array([np.array(Image.open(f)) for f in filelist_test])
		y_test = np.array([np.array(l) for l in label_test])

		return (x_train, y_train), (x_test, y_test)
	elif dataset == "FV-USM":
		path = "./Dataset/FV-USM/"
		sessions = ["1st_session/extractedvein/vein", "2nd_session/extractedvein/vein"]
		testcase = "001"
		filetype = ".jpg"
		
		filelist_train = []
		filelist_test = []
		label_train = []
		label_test = []

		for session in sessions:
			for i in range(123):
				for j in range(1, 5):
					for index in range(1, 7):
						filename = path + session + testcase + "_" + str(j) + "/0" + str(index) + filetype
						if index < 5:
							filelist_train.append(filename)
							label_train.append(testcase)
							label_train.append(testcase)
							label_train.append(testcase)
						else:
							filelist_test.append(filename)
							label_test.append(testcase)
							label_test.append(testcase)
							label_test.append(testcase)
				testcase = "0" * (3 - len(str(i + 1))) + str(i + 1)

		x_train = np.array([np.array(Image.open(f)) for f in filelist_train])
		y_train = np.array([np.array(l) for l in label_train])
		x_test = np.array([np.array(Image.open(f)) for f in filelist_test])
		y_test = np.array([np.array(l) for l in label_test])
	
		return (x_train, y_train), (x_test, y_test)
	elif dataset == "MNIST":
		return mnist.load_data()
