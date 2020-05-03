import numpy as np

def batch_generator(X, y=None, batch_size=64):
	num_records = X.shape[0]
	for i in np.arange(0, num_records, batch_size):
		start, end = i, min(i+batch_size, num_records)
		if y is not None:
			yield X[start: end], y[start: end]
		else:
			yield X[start: end]

# Return 1-D array from input N-D array (N <= 3)
def straighten(X, flow='horizontal'):
	if flow=='vertical':
		return X.T.transpose(-1, 2, 0, 1).reshape(-1, np.prod(X.shape[1:]))
	elif flow=='horizontal':
		return X.reshape(-1, np.prod(X.shape[1:]))

# Return N-D array (N <= 3) from 1-D input
def unstraighten(X, output_dim, flow='horizontal'):
	Ch, Row, Col = output_dim[0], output_dim[1], output_dim[2]
	
	if flow=='vertical':
		return X.reshape(-1, Ch, Row, Col).T.transpose(-1, 2, 0, 1)
	elif flow=='horizontal':
		return X.reshape(-1, Ch, Row, Col)
