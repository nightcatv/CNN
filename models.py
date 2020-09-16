import keras
from keras import regularizers
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization, concatenate, add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D

values = {
	"DROPOUT": 0.4,
	"LRN2D_NORM": True,
	"WEIGHT_DECAY": 0.0005,
	"DATA_FORMAT": 'channels_last'
}

def LeNet(width, height, depth, classes):
	# Initialize
	model = Sequential()

	# 1st Convolutional Layer
	model.add(Conv2D(input_shape = (width, height, depth), kernel_size = (5, 5), filters = 6, strides = (1, 1), activation = 'tanh'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

	# 2nd Convolutional Layer
	model.add(Conv2D(input_shape = (width, height, depth), kernel_size = (5, 5), filters = 16, strides = (1, 1), activation = 'tanh'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

	# Fully Connection Layer
	model.add(Flatten())
	model.add(Dense(120, activation = 'tanh'))
	model.add(Dense(84, activation = 'tanh'))

	# Output Layer - Softmax Classifier
	model.add(Dense(classes))
	model.add(Activation("softmax"))
	
	return model


def AlexNet(width, height, depth, classes):
	# Initialize
	model = Sequential()

	# 1st Convolutional Layer
	model.add(Conv2D(filters = 96, input_shape = (width, height, depth), kernel_size = (11, 11), strides = (4, 4), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# 2nd Convolutional Layer
	model.add(Conv2D(filters = 256, kernel_size = (5, 5), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# 3rd Convolutional Layer
	model.add(Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))

	# 4th Convolutional Layer
	model.add(Conv2D(filters = 384, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))

	# 5th Convolutional Layer
	model.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# Fully Connection Layer
	model.add(Flatten())

	# 1st Fully Connection Layer
	model.add(Dense(4096, input_shape = (width, height, depth), activation = 'relu'))
	model.add(Dropout(values["DROPOUT"]))		# add dropout to prevent overfitting

	# 2nd Fully Connection Layer
	model.add(Dense(4096, activation = 'relu'))
	model.add(Dropout(values["DROPOUT"]))

	# 3rd Fully Connection Layer
	model.add(Dense(1000, activation = 'relu'))
	model.add(Dropout(values["DROPOUT"]))

	# Output Layer - Softmax Classifier
	model.add(Dense(classes))
	model.add(Activation('softmax'))

	return model


def VGG16(width, height, depth, classes):
	# Initialize
	model = Sequential()

	# 1st Convolutional Layer
	model.add(Conv2D(filters = 64, input_shape = (width, height, depth), kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# 2nd Convolutional Layer
	model.add(Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# 3rd Convolutional Layer
	model.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# 4th Convolutional Layer
	model.add(Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# 5th Convolutional Layer
	model.add(Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))

	# Fully Connection Layer
	model.add(Flatten())

	# 1st Fully Connection Layer
	model.add(Dense(units = 4096, input_shape = (width, height, depth), activation = 'relu'))
	
	# 2nd Fully Connection Layer
	model.add(Dense(units = 4096, activation = 'relu'))

	# Output Layer - Softmax Classifier
	model.add(Dense(classes))
	model.add(Activation('softmax'))

	return model


def GoogLeNet_v1(width, height, depth, classes):
	# Define convolution with batchnormalization
	def Conv2D_BN(x, nb_filter, kernel_size, padding = 'same', strides = (1, 1), name = None):
		if name is not None:
			bn_name = name + '_bn'
			conv_name = name + '_conv'
		else:
			bn_name = None
			conv_name = None

		x = Conv2D(nb_filter, kernel_size, padding = padding, strides = strides, activation = 'relu', name = conv_name)(x)
		x = BatchNormalization(axis = 3, name = bn_name)(x)
		
		return x

	# Define Inception structure
	def Inception(x, nb_filter_para):
		(branch1, branch2, branch3, branch4) = nb_filter_para
		branch_1x1 = Conv2D(branch1[0], (1, 1), padding = 'same', strides = (1, 1), name = None)(x)

		branch_3x3 = Conv2D(branch2[0], (1, 1), padding = 'same', strides = (1, 1), name = None)(x)
		branch_3x3 = Conv2D(branch2[1], (3, 3), padding = 'same', strides = (1, 1), name = None)(branch_3x3)

		branch_5x5 = Conv2D(branch3[0], (1, 1), padding = 'same', strides = (1, 1), name = None)(x)
		branch_5x5 = Conv2D(branch3[1], (1, 1), padding = 'same', strides = (1, 1), name = None)(branch_5x5)

		branch_pool = MaxPooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(x)
		branch_pool = Conv2D(branch4[0], (1, 1), padding = 'same', strides = (1, 1), name = None)(branch_pool)

		x = concatenate([branch_1x1, branch_3x3, branch_5x5, branch_pool], axis = 3)

		return x
	
	innput = Input(shape = (width, height, depth))

	x = Conv2D_BN(innput, 64, (7, 7), strides = (2, 2), padding = 'same')
	x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
	x = Conv2D_BN(x, 192, (3, 3), strides = (1, 1), padding = 'same')
	x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

	x = Inception(x, [(64, ), (96, 128), (16, 32), (32, )])						# Inception 3a 28x28x256
	x = Inception(x, [(128, ), (128, 192), (32, 96), (64, )])					# Inception 3b 28x28x480
	x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)	# 14x14x480

	x = Inception(x, [(192, ), (96, 208), (16, 48), (64, )])					# Inception 4a 14x14x512
	x = Inception(x, [(160, ), (112, 224), (24, 64), (64, )])					# Inception 4a 14x14x512
	x = Inception(x, [(128, ), (128, 256), (24, 64), (64, )])					# Inception 4a 14x14x512
	x = Inception(x, [(112, ), (144, 288), (32, 64), (64, )])					# Inception 4a 14x14x528
	x = Inception(x, [(256, ), (160, 320), (32, 128), (128, )])					# Inception 4a 14x14x832
	x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)	# 7x7x832

	x = Inception(x, [(256, ), (160, 320), (32, 128), (128, )])					# Inception 5a 7x7x832
	x = Inception(x, [(384, ), (192, 384), (48, 128), (128, )])					# Inception 5b 7x7x1024

	# Using AveragePooling replace flatten
	x = AveragePooling2D(pool_size = (7, 7), strides = (7, 7), padding = 'same')(x)
	x = Flatten()(x)
	x = Dropout(0.4)(x)
	x = Dense(1000, activation = 'relu')(x)
	x = Dense(classes, activation = 'softmax')(x)

	model = Model(inputs = innput, outputs = x)

	return model


def ResNet34(width, height, depth, classes):
	# Define convolution with batchnormalization
	def Conv2D_BN(x, nb_filter, kernel_size, padding = 'same', strides = (1, 1), name = None):
		if name is not None:
			bn_name = name + '_bn'
			conv_name = name + '_conv'
		else:
			bn_name = None
			conv_name = None

		x = Conv2D(nb_filter, kernel_size, padding = padding, strides = strides, activation = 'relu', name = conv_name)(x)
		x = BatchNormalization(axis = 3, name = bn_name)(x)

		return x
	
	# Define Residual Block for ResNet34 (2 convolution layers)
	def Residual_Block(input_model, nb_filter, kernel_size, strides = (1, 1), with_conv_shortcut = False):
		x = Conv2D_BN(input_model, nb_filter = nb_filter, kernel_size = kernel_size, strides = strides, padding = 'same')
		x = Conv2D_BN(x, nb_filter = nb_filter, kernel_size = kernel_size, padding = 'same')

		# need convolution on shortcut for add different channel
		if with_conv_shortcut:
			shortcut = Conv2D_BN(input_model, nb_filter = nb_filter, strides = strides, kernel_size = kernel_size)
			x = add([x, shortcut])
			return x
		else:
			x = add([x, input_model])
			return x

	innput = Input(shape = (width, height, depth))

	x = Conv2D_BN(innput, 64, (7, 7), strides = (2, 2), padding = 'same')
	x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

	# Residual conv2_x output 56x56x64
	x = Residual_Block(x, nb_filter = 64, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 64, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 64, kernel_size = (3, 3))

	# Residual conv3_x output 28x28x128
	x = Residual_Block(x, nb_filter = 128, kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)	# need to do convolution to add different channel
	x = Residual_Block(x, nb_filter = 128, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 128, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 128, kernel_size = (3, 3))

	# Residual conv4_x output 14x14x256
	x = Residual_Block(x, nb_filter = 256, kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)	# need to do convolution to add different channel
	x = Residual_Block(x, nb_filter = 256, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 256, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 256, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 256, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 256, kernel_size = (3, 3))

	# Residual conv5_x output 7x7x512
	x = Residual_Block(x, nb_filter = 512, kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
	x = Residual_Block(x, nb_filter = 512, kernel_size = (3, 3))
	x = Residual_Block(x, nb_filter = 512, kernel_size = (3, 3))

	# Using AveragePooling replace flatten
	x = GlobalAveragePooling2D()(x)
	x = Dense(classes, activation = 'softmax')(x)

	model = Model(inputs = innput, outputs = x)

	return model

