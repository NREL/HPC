def array_multiply(a, b):
	"""Routine to take two 2D numpy arrays and do simple row-by-column accumulation
	   of new matrix elements. Return as Numpy matrix.

	   Takes two positional arguments. Both must be 2D Numpy arrays.
	   The number of columns of the first must match the number of rows in the second.
	"""
	assert np.shape(a)[1] == np.shape(b)[0]
	c = np.empty((np.shape(a)[0], np.shape(b)[1]))
	for i in range(np.shape(a)[0]):
		for j in range(np.shape(b)[1]):
			for k in range(np.shape(a)[1]):
				c[i,j] += a[i,k] * b[k,j]
	return c

def matrix_multiply(a, b):
	"""Convert input arrays to two numpy matrices and use Numpy internals to multiply."""
	assert np.shape(a)[1] == np.shape(b)[0]
	a_p = np.asmatrix(a)
	b_p = np.asmatrix(b)
	return a_p * b_p

