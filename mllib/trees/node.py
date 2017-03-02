from ..utils.Exceptions import TreeError
from collections import defaultdict

class Node(object):
	"""
	Node data structure for creating trees.
	"""
	def __init__(self,nodeId=None,child=None,root=None,maxclass=None,classDist=None):
		self.id=nodeId
		if root is None:
			self.root=self
		else:
			self.root=root
		self.child=defaultdict(str)
		if child is not None:
			for (k,v) in child.items():
				self.child[k]=v
		self.maxclass=maxclass
		self.classDist=classDist

	def getChild(self,element):
		# Return the chhild with the given identifier
		if element not in self.child.keys():
			raise TreeError("Invalid child identifier, ",element, "Available : ", "".join(self.child.keys()))
		return(self.child[child])

	def setChild(self,element,child):
		# Set the child to the given attribute
		self.child[element]=child

	def getAllChildren(self):
		# Return All the children
		return(self.child)

	def setAllChildren(self,childrens):
		# Set Childrens
		if child is not None or child == {}:
			self.child=defaultdict(str)
			for (k,v) in childrens.items():
				self.child[k]=v
		else:
			raise TreeError("Invalid Tree Inputs")

	def getRoot(self):
		# Returns the root of ree,
		return(self.root)

class TreeFunctions:
	"""
	Pre-Order Tree Traversal functions
	"""
	def traverseFromNode(self,node,splitAt=None,level=0):
		"""
		Given the root node, does a pre-order traversal and prints the nodes.
		Stops at leaf nodes.
		"""
		print("|\t" * (level),splitAt,node.id,node.classDist,node.maxclass)
		children=node.getAllChildren().items()
		if len(children) > 0:
			for en,i in enumerate(children):
				self.traverseFromNode(i[1],i[0],level+1)