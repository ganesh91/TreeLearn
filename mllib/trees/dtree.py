from ..trees.node import Node
from ..core.core import Estimator,ClassifierTrait
from ..trees.entropy import singleAttributeEntropy,twoAttributeEntropy,informationGain,maxGainFeature
from ..utils.Exceptions import UnFittedModelError,FittedModelError,UnImplementedError,DTreeError
from ..utils.Utils import countClass,likelyClass
from collections import defaultdict

class DTreeClassifier(Estimator,ClassifierTrait):
	"""
	Decision Tree Classifier based on Entropy. Works only on categorical variables.
	DTreeClassifier is initialized with the parameters when the class is instantiated
	and functions fit is used to fit the data to the dtree. Predict function returns the
	list of predictions for every feature vector. Score internally calls predict, uses the
	actual outcome and real outcome to create a confusion matrix.
	"""
	def __init__(self,depth=1,verbose=False):
		self.depth=depth
		self.verbose=verbose
		self.isFitted=False
		self.model=None
		self.parameters={'depth':depth,'verbose':verbose}

	def _growTree(self,_class,features,tree=None,featureLevel=None,level=0,data=None):
		"""
		Grows the decision tree. Initially tree is None, Hence the root node is initialized
		as the max Information Gain feature and recursively builds the whole tree from there on.
		"""
		# If level < possible depth and features are atleast true, Run
		if level <= self.depth and len(features) > 0:
			# Given no data, consume the whole training data. else use the filtered data.
			if data is None:
				df=self.traindata
			else:
				df=data
			# Find the Most Significant Feature to split at.
			_maxGainFeature=maxGainFeature(df,features,_class,self.verbose)
			# Find possible discrete values of the most significant feature
			naryLevels=df.select([_maxGainFeature]).getColumnLevel()

			#Class Counts before split:
			_classCountsBef=countClass(df.select([_class]).getColumnLevels())
			_likelyClassBef=likelyClass(_classCountsBef)

			#If There if no Tree, Generate a Node
			if tree is None:
				tree=Node(_maxGainFeature,maxclass=_likelyClassBef,classDist=_classCountsBef)
			else:
				#If there is a tree, grow the tree by creating the node.
				child=Node(_maxGainFeature,root=tree.getRoot(),maxclass=_likelyClassBef,classDist=_classCountsBef)
				tree.setChild(featureLevel,child)
				tree=child

			#For all possible feature level in nary features of max gain attribute, recurse and split further
			#till you reach the given depth or till you cannot split more.
			for featureLevel in naryLevels:
				if self.verbose:
					print(_maxGainFeature,featureLevel)
				if len(df.select([_class]).getColumnLevel()) > 1:
					newFeatures= [feature for feature in features if feature != _maxGainFeature]
					self._growTree(_class,newFeatures,tree,featureLevel,level+1,
						df.boolean_index([(_maxGainFeature,'=',featureLevel)]))
		else:
			pass
		self.model=tree.getRoot()
		self.isFitted=True

	def _traversePredict(self,verbose):
		"""
		Traverse the d-tree and predict the most possible label for the feature vector.
		"""
		metadata=[k for k,v in sorted(self.testdatset.metadata.items(), key=lambda x: x[1])]
		row_prediction=[]
		for row in self.testdatset.data:
			rowdict=dict(zip(metadata,row))
			likelyClass=None
			traverse_stack=[self.model]
			# Do a Breadth First like traversal of the tree
			while(len(traverse_stack)>0):
				node=traverse_stack.pop()
				if len(node.child) > 0 and node.child[rowdict[node.id]] != "":
					if verbose:
						print(node.id,
							rowdict[node.id],
							node.child[rowdict[node.id]].id,
							rowdict)
					# If has a further node, append the node to the list.
					traverse_stack.append(node.child[rowdict[node.id]])
				else:
					# If there is no further node, use the max class at that
					# point in time as the most likely class.
					likelyClass=node.maxclass
			row_prediction.append(likelyClass)
		return(row_prediction)

	def fit(self,df,_class=None,skip=[]):
		"""
		Create a decision tree of the by fitting to the given data frame.
		"""
		if self.isFitted:
			raise FittedModelError("Attempting to fit a already fitted model")
		if _class is None:
			raise DTreeError("_class is empty")
		elif _class not in df.metadata.keys():
			raise DTreeError(_class," is not a feature in the dataset")
		self.traindata=df
		self._class=_class
		features=[k for k in self.traindata.metadata.keys() if k != _class and k not in skip and k[0]!='_']
		self._features=features
		self._growTree(_class,features,None)
		return(self)

	def predict(self,df,verbose=False):
		"""
		Predict the outcomes for given dataset.
		"""
		if not self.isFitted:
			raise UnFittedModelError("Attempting to predict on a unfitted model")
		self.testdatset=df
		return(self._traversePredict(verbose))


	def _confusionMatrix(self,x_actual,x_pred):
		"""
		Create Confusion matrix dataframe.
		"""
		cm=super(DTreeClassifier,self)._confusionMatrix(x_actual,x_pred)
		return(cm)

	def score(self,df,verbose=False):
		"""
		Call predict and compare actual vs predicted.
		"""
		actual_outcome=list(df.select([self._class]).getColumnLevels())
		validationdf=df.select(self._features)
		pred_outcomes=self.predict(validationdf,verbose)
		cm=self._confusionMatrix(actual_outcome,pred_outcomes)
		cm.show(9999)
