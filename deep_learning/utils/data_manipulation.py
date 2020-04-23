import numpy as np

def batch_generator(X, y=None, batch_size=64):
	num_records = X.shape[0]
	for i in np.arange(0, num_records, batch_size):
		start, end = i, min(i+batch_size, num_records)
		if y is not None:
			yield X[start: end], y[start: end]
		else:
			yield X[start: end]

def straighten(X, flow='vertical'):
	if flow=='vertical':
		return X.T.flatten()
	elif flow=='horizontal':
		return X.flatten()
	else:
		raise ValueError('keyword: ' + flow + ' not recognized')
