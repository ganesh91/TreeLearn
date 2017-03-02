def zeroes(x_rows,y_columns):
	"""
	Create a ndarray of zeroes of x rows and y columns
	"""
	ndarray=[]
	for _ in range(x_rows):
		row=[]
		for _ in range(y_columns):
			row.append(0)
		ndarray.append(row)
	return(ndarray)

