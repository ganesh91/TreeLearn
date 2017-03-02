from ..core.core import Estimator,ClassifierTrait
from ..utils.Exceptions import UnImplementedError,BoostingException
from ..utils.EnsembleMetrics import calculateVote,getMode

class votingClassifier(Estimator,ClassifierTrait):
	"""
	Basic Voting Classifier. Given several estimators, the predictions are collected and voted 
	using the given vote method. By default, majority voting is used, i.e mode of the predictions
	are returned as the output. However, Min, Max, random, weighted voting can be done using
	developing appropriate weighting functions.
	"""
	def __init__(self,models,voteMethod=getMode):
		self.models=models
		self.voteMethod=voteMethod

	def predict(self,testdata,kwargs={}):
		predictions=[]
		pairwise_predictions=[]
		for caption,model in self.models:
			predictions.append(model.predict(testdata,**kwargs))

		for i in range(testdata.shape[0]):
			row_predictions=[]
			for pred in predictions:
				row_predictions.append(pred[i])
			pairwise_predictions.append(row_predictions)

		return(calculateVote(pairwise_predictions,self.voteMethod))


	def _confusionMatrix(self,x_actual,x_pred):
		"""
		Create Confusion matrix dataframe
		"""
		cm=super(votingClassifier,self)._confusionMatrix(x_actual,x_pred)
		return(cm)

	def fit(self,df,_class=None):
		for caption,models in self.models:
			models.fit(df,_class)

class BoostingClassifier(Estimator,ClassifierTrait):
	def __init__(self,Classifier,numOfEstimators,LossFunction):
		raise(UnImplementedError("Not Implemented"))

	def _confusionMatrix(self,x_actual,x_pred):
		"""
		Create Confusion matrix dataframe
		"""
		cm=super(BoostingClassifier,self)._confusionMatrix(x_actual,x_pred)
		return(cm)

	def _boost(self):
		"""
		performs boosting and returns model and model confidence.
		"""

	def fit(self,df,_class=None):
		raise(UnImplementedError("Not Implemented"))


	def predict(self,df):
		raise(UnImplementedError("Not Implemented"))		

		
