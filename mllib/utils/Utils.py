from collections import defaultdict
from math import ceil
from random import sample

def countClass(onedarray):
	"""
	Given a list, return a dictionary of unique labels and their counts.
	"""
	classCounts=defaultdict(int)
	for label in onedarray:
		classCounts[str(label)]+=1
	return(dict(classCounts))

def likelyClass(classcounts):
	"""
	Given a dictionary of labels and counts, return the max count label
	"""
	return(sorted(classcounts.items(),key=lambda x:x[1])[-1][0])

def test_train_split(df,split=0.2,verbose=False):
	"""
	Split the data frame into training and test by the given ratio.
	Samples random numbers from 0 to # of rows and creates True or False
	boolean filter and uses these boolean filter to filter the data.
	"""
	rows=df.shape[0]
	test_proportion=ceil(split*rows)
	train_proportion=rows-test_proportion
	if verbose:
		print("Test - Train Split Ratio",split)
		print("# of Rows in Original Dataset",rows)
		print("# of Rows in Train Dataset",train_proportion)
		print("# of Rows in Test Dataset",test_proportion)
	test_rowids=set(sample(range(rows),test_proportion))
	test_boolean=[]
	train_boolean=[]
	for i in range(rows):
		if i in test_rowids:
			test_boolean.append(True)
			train_boolean.append(False)
		else:
			train_boolean.append(True)
			test_boolean.append(False)
	traindf=df.select(None,train_boolean)
	testdf=df.select(None,test_boolean)
	return((traindf,testdf))


			