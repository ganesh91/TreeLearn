from statistics import mode,StatisticsError
from random import choice
from math import exp,log,sqrt
from ..utils.Exceptions import BoostingException,AccuracyLessThanRandom

def getMode(vector):
	"""
	Given a list of elements, returns the mode of the list.
	If the all elements in the list are unique, returns a random value.
	"""
	# print(vector)
	try:
		return(mode(vector))
	except StatisticsError:
		return(choice(vector))

def calculateVote(listoflists,vote_function=getMode,kwargs={}):
	"""
	Base wrapper for vote calculation. Calls the selected voting method, argmax by default
	or the selected voting function and returns the results.
	Input: Lists of lists, seach sublist listing the prection of same data point by multiple
	classifiers / Regressors.
	"""
	argmax_votes=[vote_function(vector,**kwargs) for vector in listoflists]
	return(argmax_votes)

def calculateExponentialLossParams(Y_actual,Y_predicted,weights):
	"""
<<<<<<< HEAD
	Calculates the exponential loss parameters. Given the current weight,
	actual and predictions, calculates eps as,
	eps = sum,forall_i(weights,Y_actual!=Y_predicted)
	alpha=0.5 log (1-eps/eps)
	normalization zk = 2 * sqrt(eps(1-eps))
	source: http://math.mit.edu/~rothvoss/18.304.3PM/Presentations/1-Eric-Boosting304FinalRpdf.pdf
	Returns, eps,alpha,normalization constant
=======
	Calculates the exponential loss parameters, checks if the predicted outcome = actual outcome,
	if not adds the weight to the eps. alpha, and normalization z is also calculated.
>>>>>>> origin/master
	"""
	if len(Y_actual) != len(Y_predicted) != len(weights):
		raise(BoostingException("The Length of Actual, Precicted and Weights doesn't match"))

	eps=0
	for i in range(len(Y_actual)):
		if Y_actual[i] != Y_predicted[i]:
			eps+=weights[i]

	alpha=0.5*log((1-eps)/eps)
	if alpha <= 0 :
		raise(AccuracyLessThanRandom("Alpha for boosting cannot be less than zero, got "+str(alpha)))
	norm_cnst=2*sqrt(eps*(1-eps))

	return((eps,alpha,norm_cnst))

def exponentialLoss(Y_actual,Y_predicted,weights):
	"""
<<<<<<< HEAD
	Given a Y_actual, Y_predicted and weights, returns the updated weights and alpha.
	By Default, Implements the weighting measure of AdaBoost.
=======
	Given Actual, Predicted and Weights of the current iteration, calculates the updated weights and
	the model confidence alpha. Internally calls calculateExponentialLossParams to calculate the params.
>>>>>>> origin/master
	"""
	try:
		eps,alpha,norm_cnst = calculateExponentialLossParams(Y_actual,Y_predicted,weights)
		functionPositive = exp(-1*alpha)
		functionNegative = exp(alpha)
		updatedWeights=[]

		for i in range(len(Y_actual)):
			if Y_actual[i] == Y_predicted[i]:
				updatedWeights.append(weights[i]*functionPositive)
			else:
				updatedWeights.append(weights[i]*functionNegative)

		return((updatedWeights,alpha))
	except AccuracyLessThanRandom as e:
		raise e

def resolveSign(predictions,modelWeights):
	"""
	Given a set of model confidences, and predictions, calculates the hypothesis of outcome of a data point.
	"""
	if len(predictions) != len(modelWeights):
		raise((BoostingException("The length of predictions and Model Weights does not Match")))
	h0=0

	for i in range(len(predictions)):
		h0+=predictions[i]*modelWeights[i]
	if h0 >= 0:
		h0 = 1
	else:
		h0 = -1
	return(h0)
