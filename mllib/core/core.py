from ..datastructures.DataFrame import dataframe
from ..metrics.ConfusionMatrix import confusionMatrix,accuracy
from ..utils.Exceptions import UnImplementedError

class Estimator(object):
	"""
	Base Class for All estimators
	"""
	def __init__(self,**kwars):
		self.parameters=kwargs

	def printParamters(self):
		for (k,v) in self.parameters.items():
			print(k," : ",v)

class ClassifierTrait(object):
	"""
	Mix-in Trait froa all classifiers
	"""
	def score(self,actual,predicted):
		cm=confusionMatrix(actual,predicted)
		return(accuracy)

	def _confusionMatrix(self,actual,predicted):
		return(confusionMatrix(actual,predicted))

	def fit(self,df,_class=None):
		raise UnImplementedError("The function is not implemented yet.")

	def predict(self,df):
		raise UnImplementedError('The function is not implemented yet')

		
