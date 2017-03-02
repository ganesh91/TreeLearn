from ..ensemble.EnsembleBase import votingClassifier
from ..core.core import Estimator,ClassifierTrait
from ..trees.dtree import DTreeClassifier
from ..utils.Exceptions import UnFittedModelError,FittedModelError,UnImplementedError,DTreeError
import multiprocessing as mp

class RandomForestClassifier(Estimator,ClassifierTrait):
	"""
	Creates a Random Forest Classifier. Change the constructor available for serial and
	parallel execution.
	"""
	def __init__(self,forestSize=10,depth=1,forestPopulation='actual',njobs=1):
		self.forestSize=forestSize
		self.depth=depth
		self.forestPopulation=forestPopulation
		self.isFitted=False
		self.model=None
		self.njobs=njobs
		self.parameters={'forestSize':forestSize,'depth':depth,'forestPopulation':forestPopulation
						,'njobs':njobs}

	def wrapDTree(self,kwags):
			return(DTreeClassifier(depth=self.depth,**self.kwClfMetaParams)).fit(**kwags)

	def mergeDict(self,dict1,dict2):
			z=dict1.copy()
			z.update(dict2)
			return(z)

	def fit(self,traindata,_class,kwClfMetaParams={},kwClfParams={}):
		if self.isFitted:
			raise FittedModelError("Attempting to fit a already fitted model")
		if _class is None:
			raise DTreeError("_class is empty")
		elif _class not in traindata.metadata.keys():
			raise DTreeError(_class," is not a feature in the dataset")

		models=[]
		self._class=_class
		self.kwClfMetaParams=kwClfMetaParams

		if self.njobs == 1:
			for num in range(self.forestSize):
				dtreeClf=DTreeClassifier(depth=self.depth,**kwClfMetaParams)
				if self.forestPopulation == 'actual':
					dtreeClf.fit(traindata.sample(traindata.shape[0],True),_class,**kwClfParams)
				else:
					raise(UnImplementedError("Currently supports only actual size"))
				models.append((num,dtreeClf))
		elif self.njobs > 1:

			processes=[]

			bootstraps = [{'df':traindata.sample(traindata.shape[0],True)
							,'_class':self._class} for i in range(self.forestSize)]

			loopDicts = [self.mergeDict(i,kwClfParams) for i in bootstraps]

			runInstances=mp.Pool(self.njobs).map(self.wrapDTree, loopDicts)
			#ToDo : Update the Counter Later
			models=[(1,i) for i in runInstances]

		vc = votingClassifier(models)

		self.model = vc
		self.isFitted=True

	def predict(self,testdf,kwargs={}):
		"""
		"""
		if not self.isFitted:
			raise UnFittedModelError("Attempting to predict on a unfitted model")

		return(self.model.predict(testdf,kwargs))

	def _confusionMatrix(self,x_actual,x_pred):
		"""
		Create Confusion matrix dataframe
		"""
		cm=super(RandomForestClassifier,self)._confusionMatrix(x_actual,x_pred)
		return(cm)

	def score(self,df,kwargs={'verbose':False}):
		"""
		Call predict and compare actual vs predicted
		"""
		actual_outcome=list(df.select([self._class]).getColumnLevels())
		validationdf=df.select([feature for feature in df.metadata.keys() if feature != self._class])
		pred_outcomes=self.predict(validationdf,kwargs)
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
