import matplotlib.pyplot as plt

# Plot loss and accuracy
def Plot(History):
	plt.figure(figsize = (15, 5))
	plt.subplot(1, 2, 1)
	plt.plot(History.history['accuracy'])
	plt.plot(History.history['val_accuracy'])
	plt.title('Model Accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')

	plt.subplot(1, 2, 2)
	plt.plot(History.history['loss'])
	plt.plot(History.history['val_loss'])
	plt.plot('Model Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc = 'upper left')

	plt.show()
	plt.show()
