from itertools import chain
from math import log

def singleAttributeEntropy(df,_class,verbose=False):
	"""
	Calculates single class entropy. The data frame passed should have only
	one column, i.e list of lists.
	"""
	if verbose:
		print("Incoming Dataframe Shape",df.shape)
	totalCount=df.shape[0]
	classes=df.select([_class]).getColumnLevel()
	if verbose:
		print("Types of classes",classes)
	entropy=0
	for cl in classes:
		tmp=df.boolean_index([(_class,'=',cl)])
		classCount=sum(tmp.select(["_weight"]).getColumnLevels())
		if verbose:
			print("The class ",cl," number ",tmp.shape)
		#Entropy formula
		entropy -= (classCount/totalCount)*log(classCount/totalCount,2)
	return(entropy)

def twoAttributeEntropy(df,attribute,_class,verbose=False,level='total'):
	"""
	Given two columns, column A and a _class (outcome), calculates mutual
	information.
	"""
	totalCount=df.shape[0]
	classes=df.select([attribute]).getColumnLevel()
	entropy=0
	for cl in classes:
		tmp=df.boolean_index([(attribute,'=',cl)])
		classCount=sum(tmp.select(["_weight"]).getColumnLevels())
		classEntropy=singleAttributeEntropy(tmp,_class)
		if verbose and level=='fine':
			print("Class Probability",cl,"Total",totalCount,"CC",classCount,"Ratio",classCount/totalCount)
			print("Single Entropy of Class",classEntropy)
		entropy +=(classCount/totalCount)*classEntropy
	if verbose and level=='total':
		print("Total Entropy for column",attribute,"is",entropy)
	return(entropy)

def informationGain(classEntropy,MutualInformation):
	"""
	Given class entropy and MutualInformation, calculates Information Gain
	"""
	return(classEntropy-MutualInformation)

def maxGainFeature(traindata,features,_class,verbose):
	"""
	Calculates Entropy of class, mutual information for all possible features,
	and calculates information gain for the set of features.
	Returns the feature which has the most information gain.
	"""
	#Calculate Class Entropy
	classEntropy=singleAttributeEntropy(traindata,_class,verbose)
	selector=[]
	#Calculate Mutual Information for every feature
	for feature in features:
		featureEntropy=twoAttributeEntropy(traindata,feature,_class,verbose)
		selector.append((feature,informationGain(classEntropy,featureEntropy)))
	#Return the Maximum value
	maxGainFeature=sorted(selector,key = lambda x: x[1])[-1]
	return(maxGainFeature[0])
