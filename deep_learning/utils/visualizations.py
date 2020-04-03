import numpy as np
import matplotlib.pyplot as plt

class plot_curves():
	def __call__(self, history, type_):
		if type_ == 'cost':
			# Plot cost vs epoch
			plt.plot(history['cost'])
			plt.title('Model accuracy')
			plt.ylabel('Cost')
			plt.xlabel('Epoch')
			plt.legend(['Cost'], loc='upper left')
			plt.show()
