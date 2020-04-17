import numpy as np
import matplotlib.pyplot as plt

class plot_curves():
	def __call__(self, history, type_):
		if type_ == 'loss':
			# Plot cost vs epoch
			plt.plot(history['cost'])
			plt.title('Cost vs Epoch')
			plt.ylabel('Cost')
			plt.xlabel('Epoch')
			plt.legend(['Cost'], loc='upper left')
			plt.show()

		if type_ == 'valid':
			legend_list = []

			for key in history.keys():
				if key != 'cost':
					plt.plot(history[key])
					legend_list.append(key)

			plt.title('Errors vs Epoch')
			plt.ylabel('Errors')
			plt.xlabel('Epoch')
			plt.legend(legend_list, loc='upper left')
			plt.show()
