"""
Exception classes used in the mllib.
"""
class IllegalDataFrameStateError(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)

class ScoringError(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)

class TreeError(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)

class UnImplementedError(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)

class UnFittedModelError(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)	

class FittedModelError(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)	

class DTreeError(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)	

class BoostingException(Exception):
	def __init__(self,message):
		self.message=message
		print(self.message)	

class AccuracyLessThanRandom(BoostingException):
	def __init__(self,message):
		self.message=message

