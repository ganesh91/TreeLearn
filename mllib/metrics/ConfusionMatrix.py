from ..datastructures.DataFrame import dataframe
from ..utils.Exceptions import ScoringError,UnImplementedError
from ..utils.Zeroes import zeroes

from decimal import Decimal

def confusionMatrix(y_actual,y_predicted):
	"""
	Given list of actuals and predicted, create confusion Matrix as a
	data frame object and return the dat fram object.
	"""
	if len(y_actual) - len(y_predicted) != 0 :
		raise ScoringError("Mismatch in # of observations in actual and predicted. Expected ",len(y_actual), " observed",len(y_predicted))
	for i in range(len(y_actual)):
		if isinstance(y_actual[i],Decimal) or isinstance(y_predicted[i],Decimal):
			raise ScoringError("Confusion Matrix cannot be created for numbers")
	classes=sorted(set(y_actual).union(y_predicted))
	cm=zeroes(len(classes),len(classes))
	classdict=dict(zip(classes,range(len(classes))))
	for actual,pred in zip(y_actual,y_predicted):
		cm[classdict[actual]][classdict[pred]]+=1
	for _class,index in sorted(classdict.items(),key=lambda x: x[1]):
		cm[index].insert(0,_class)
	metadata=list(classes)
	metadata.insert(0,"")
	title=dict(zip(metadata,range(len(metadata))))
	return dataframe(metadata=title,data=cm)

def accuracy(cm):
	"""
	Given the confusion matrix, return accuracy.
	"""
	header=[x for x in cm.metadata.keys() if len(x) > 0]
	cm_1=cm.select(sorted(header))
	cm_1.show()
	total=0
	diagonal=0
	for i in range(len(header)):
		for j in range(len(header)):
			total+=cm_1.data[i][j]
			if i==j:
				diagonal+=cm_1.data[i][j]
	return(diagonal/total)