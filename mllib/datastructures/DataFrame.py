from collections import defaultdict
from ..utils.Exceptions import IllegalDataFrameStateError
from itertools import chain
from copy import deepcopy
from random import sample,choice

class dataframe:
	"""
	Data Frame API inspired from Pandas and Numpy.
	Data Frame class consists of two python objects,
	1. Metadata which consists of a dict {'Columns': Index location in list}
	2. Data : List of lists, every list is a row [[a,b,c],[d,e,f],[g,h,i]]

	Base Functions include:
	1. Slicing Columns : select()
	2. Filtering Rows : booleanfilter()
	"""
	def __init__(self,metadata=None,data=None):
		if metadata is not None or data is not None:
			if len(metadata.keys()) != len(data[0]):
				raise IllegalDataFrameStateError("Metadata length doesn't match with data")
		self.metadata=metadata
		self.data=data
		self.shape=self.updateShape()

	def updateShape(self):
		shape=(0,0)
		if self.data==None:
			shape=(0,0)
		elif self.data==[]:
			shape=(0,0)
		else:
			shape=(len(self.data),len(self.data[0]))
		return(shape)

	def read_csv(self,filename,delimiter=",",header=None):
		"""
		Read a CSV file to data frame.
		"""
		with open(filename) as fileObject:
			if header is None:
				header=dict([(text,index) for (index,text) in enumerate(fileObject.readline().strip().split(delimiter))])
			else:
				header=dict(zip(header,range(len(header))))
			ndarray=[]
			for line in fileObject.readlines():
				ndarray.append(line.strip().split(delimiter))
			if len(header.items()) != len(ndarray[0]):
				raise IllegalDataFrameStateError("The Given Header Length and Actual columns doesn't match.")
			self.metadata=header
			self.data=ndarray
			self.shape=self.updateShape()
		return(self)

	def to_csv(self,filename,delimiter=","):
		"""
		Writes a dataframe as csv file
		"""
		with open(filename,'w') as fl:
			fl.write(delimiter.join([k for k,v in sorted(self.metadata.items(),key=lambda  x: x[1])])+"\n")
			for row in self.data:
				fl.write(",".join([str(element) for element in row]))
				fl.write("\n")

	def show(self,lines=5):
		"""
		Prints first 5 rows of the data frame.
		"""
		print(",".join([head for head,index in sorted(self.metadata.items(),key=lambda x: x[1])]))
		for en,line in enumerate(self.data):
			if en < lines:
				print(",".join([str(li) for li in line]))
		return(self)

	def _checkColumns(self,columns):
		for column in columns:
			if column not in self.metadata.keys():
				raise IllegalDataFrameStateError("The column \""+str(column)+"\" doesn't exist.")

	def select(self,columns=None,boolean_filter=None):
		"""
		Selects and returns columns from the data.
		By default selects all columns. If a list is passed to columns, only those columns are selected.
		If a Boolen Filter is present only those rows having a true is selected.
		"""
		columns=list(self.metadata.keys()) if columns is None else columns
		if boolean_filter is None:
			boolean_filter=[True for i in range(len(self.data))]
		if len(boolean_filter) != len(self.data):
			raise IllegalDataFrameStateError("The boolean Filter length "+str(len(boolean_filter))+" doesn't match data size ",len(self.data))
		self._checkColumns(columns)
		if len(self.data)==0 or len(self.data[0])==0:
			raise IllegalDataFrameStateError("The empty data frame cannot be selected")
		if len(columns)==1:
			ndarray=[]
			index=self.metadata[columns[0]]
			for en,rows in enumerate(self.data):
				if boolean_filter[en]:
					ndarray.append([rows[index]])
			return(dataframe(metadata={columns[0]:0},data=ndarray))
		if len(columns)>1:
			ndarray=[]
			for en,row in enumerate(self.data):
				if boolean_filter[en]:
					nrow=[]
					for column in columns:
						nrow.append(row[self.metadata[column]])
					ndarray.append(nrow)
			return(dataframe(dict(zip(columns,range(len(columns)))),ndarray))

	def boolean_index(self,conditions,columns=None):
		"""
		creates boolean index for given conditions.
		Conditions are typically given as [('column','relation','condition'),('column','relation','condition')]
		"""
		boolColumns=[column for column,operator,operand in conditions]
		self._checkColumns(boolColumns)
		boolMatCollection=[]
		for condition in conditions:
			column,operator,operand=condition
			if operator not in ['=','<','>']:
				raise IllegalDataFrameStateError("Filters can only be in =,<,>")
			nrow=[]
			for row in self.select([column]).data:
				if operator=="=":
					nrow.append(row[0]==operand)
				elif operator=='<':
					nrow.append(row[0]<operand)
				elif operator=='>':
					nrow.append(row[0]>operand)
			boolMatCollection.append(nrow)
		if len(conditions)>1:
			booleanFilter=[]
			for row in range(len(boolMatCollection[0])):
				condition=True
				for conditionIndex in range(len(boolMatCollection)):
					condition=condition and boolMatCollection[conditionIndex][row]
				booleanFilter.append(condition)
			boolMatCollection=booleanFilter
		else:
			boolMatCollection=boolMatCollection[0]
		return(self.select(columns,boolMatCollection))

	def getColumnLevel(self):
		"""
		If the data frame is a single column Series, returns set of discrete levels
		"""
		if len(self.metadata) > 1:
			raise IllegalDataFrameStateError("Column Level function can be performed only in a series dataframe")
		return(set(chain.from_iterable(self.data)))

	def getColumnLevels(self):
		"""
		If the data frame is a single column series, returns collapses
		list of lists to a single list and returns the same.
		"""
		if len(self.metadata) > 1:
			raise IllegalDataFrameStateError("Column Level function can be performed only in a series dataframe")
		return(chain.from_iterable(self.data))

	def sample(self,n=10,replace=False):
		"""
		Sampling the data frame with and without replacement.
		If sampled with replacement, choice is used to pick a random index everytime,
		else, python sample function is used for sampling the indexes.
		Returns a data frame with sampled data.
		"""
		sample_index=[]
		sampled_data=[]

		#Construct indexes
		if replace:
			ranges=list(range(self.shape[0]))
			sample_index=[choice(ranges) for i in range(n)]
		else:
			sample_index=sample(range(self.shape[0]),n)

		#Construct and return the data frame
		for row_id in sample_index:
			sampled_data.append(deepcopy(self.data[row_id]))

		return(dataframe(self.metadata,sampled_data))

	def updateColumn(self,column,update):
		"""
		Given a column and update list, updates the original data frame
		"""
		if len(self.data)!=len(update):
			raise(IllegalDataFrameStateError("The Update Column Size is not as same as the original data."))
		for i in range(len(update)):
			self.data[i][self.metadata[column]]=update[i]

	def createColumn(self,colName,colData):
		"""
		Appends the value of the data to the data frame. If the column is present, updates the data.
		"""
		if len(colData)!= len(self.data):
			raise(IllegalDataFrameStateError("length of the incoming data doesn't match the existing data"))
		if colName in self.metadata.keys():
			self.updateColumn(colName,colData)
		else:
			insertKey=sorted(self.metadata.items(),key=lambda x:x[1])[-1][1]+1
			self.metadata.update({colName:insertKey})
			for i in range(len(colData)):
				self.data[i].append(colData[i])
			self.shape=self.updateShape()
