# Import custom libraries
from sys import argv
from mllib.datastructures.DataFrame import dataframe
from mllib.trees.dtree import DTreeClassifier
from mllib.trees.node import TreeFunctions
from mllib.utils.Utils import test_train_split

#Read the training file from first argument
df=dataframe().read_csv(argv[1],delimiter=" ",header=['class','a1','a2','a3','a4','a5','a6','id'])
#Read the test file from second argument
tdf=dataframe().read_csv(argv[2],delimiter=" ",header=['class','a1','a2','a3','a4','a5','a6','id'])
#Initialize a dree classifier with given depth
dtree=DTreeClassifier(depth=int(argv[3]),verbose=False)
#Fit the classifier
df.createColumn('_weight',[1 for i in range(df.shape[0])])
dtree.fit(df,'class',['id'])
tfn=TreeFunctions()
#Traverse and print tree
tfn.traverseFromNode(dtree.model)
#Print confusion Matrix
dtree.score(tdf,verbose=False)
