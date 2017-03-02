from ..trees.dtree import DTreeClassifier
from ..ensemble.EnsembleBase import BoostingClassifier
from ..utils.EnsembleMetrics import exponentialLoss,resolveSign
from ..utils.Exceptions import BoostingException,AccuracyLessThanRandom

class AdaBoost(BoostingClassifier):
	def __init__(self,Classifier=DTreeClassifier,numOfEstimators=5,LossFunction=exponentialLoss):
		"""
		Initialize the classifier
		"""
		self.metaClassifier=Classifier
		self.numOfEstimators=numOfEstimators
		self.LossFunction=LossFunction
		self.isFitted=False
		self.model=None
		self.parameters={'BoostedTrees':numOfEstimators,'LossFunction':LossFunction}

	def _boost(self,clf,traindata,kwClfParams):
		"""
		Performs a Boosting Step. A boosting step involves training a classifier,
		create predictions on the train dataset, use these to calculate loss
		"""
		try:
			weights=list(traindata.select(["_weight"]).getColumnLevels())
			clf.fit(traindata,self._class,**kwClfParams)
			Y_pred=clf.predict(traindata.select(clf._features))
			Y_actual=list(traindata.select([clf._class]).getColumnLevels())
			weights=list(traindata.select(["_weight"]).getColumnLevels())
			updatedWeights,alpha=self.LossFunction(Y_actual,Y_pred,weights)
			self.traindata.updateColumn("_weight",updatedWeights)
			return((alpha,clf))
		except AccuracyLessThanRandom as e:
			raise e

	def fit(self,traindata,_class,kwClfMetaParams={},kwClfParams={}):
		"""
		Given a traindata, class and class parameters, fit the model.
		"""
		if self.isFitted:
			raise FittedModelError("Attempting to fit a already fitted model")
		if _class is None:
			raise BoostingException("_class is empty")
		elif _class not in traindata.metadata.keys():
			raise BoostingException(_class+" is not a feature in the dataset")

		models=[]
		model_confidence=[]
		self._class=_class
		self.traindata=traindata
		try:
			for num in range(self.numOfEstimators):
				estimator=self.metaClassifier(**kwClfMetaParams)
				alpha,model=self._boost(estimator,traindata,kwClfParams)
				model_confidence.append(alpha)
				models.append(model)
		except AccuracyLessThanRandom:
			pass
		finally:
			self.model=models
			self.model_confidence=model_confidence
			self.isFitted=True

	def predict(self,traindata,verbose=False):
		"""
		Given the testdata, predict.
		"""
		if not self.isFitted:
			raise UnFittedModelError("Attempting to predict on a unfitted model")

		modelResults=[]
		ensembleResult=[]
		for model in self.model:
			modelResults.append(model.predict(traindata))
		for i in range(len(modelResults[0])):
			tmpResult=[]
			for j in range(len(modelResults)):
				tmpResult.append(int(modelResults[j][i]))
			ensembleResult.append(str(resolveSign(tmpResult,self.model_confidence)))
		return(ensembleResult)

	def _confusionMatrix(self,x_actual,x_pred):
		"""
		Create Confusion matrix dataframe
		"""
		cm=super(AdaBoost,self)._confusionMatrix(x_actual,x_pred)
		return(cm)

	def score(self,df,kwargs={'verbose':False}):
		"""
		Call predict and compare actual vs predicted
		"""
		actual_outcome=list(df.select([self._class]).getColumnLevels())
		pred_outcomes=self.predict(df)
		cm=self._confusionMatrix(actual_outcome,pred_outcomes)
		print(self.accuracy(cm))

	def accuracy(self,cm):
		"""
		Calculate the accuracy of the model
		"""
		if cm.shape[0] != cm.shape[1]-1:
			raise(BoostingException("The shape of the Confusion Matrix doesn't match"))
		rsum=0
		diagsum=0
		for i in range(len(cm.data)):
			for j in range(1,cm.shape[0]+1):
				if i==(j-1):
					diagsum+=cm.data[i][j]
				rsum+=cm.data[i][j]
		return(diagsum/rsum)
