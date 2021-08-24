def logistic_function(x, r):
	f = r * x * (1-x)
	return f

def iterate_f(it, x, r):
	l = []
	for i in range(it):
		x = logistic_function(x, r)
		l += [x]
	return l
