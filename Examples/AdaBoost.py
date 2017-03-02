from sys import argv
from mllib.datastructures.DataFrame import dataframe
from mllib.utils.Utils import test_train_split
from mllib.trees.dtree import DTreeClassifier
from mllib.ensemble.AdaBoost import AdaBoost

traindf=dataframe().read_csv("agaricuslepiotatrain1.csv",delimiter=",")
testdf=dataframe().read_csv("agaricuslepiotatest1.csv",delimiter=",")

CLASS='bruises?-bruises'

traindf.updateColumn(CLASS,['1' if i == '1' else '-1' for i in traindf.select([CLASS]).getColumnLevels()])
testdf.updateColumn(CLASS,['1' if i == '1' else '-1' for i in testdf.select([CLASS]).getColumnLevels()])

traindf.createColumn('_weight',[1/traindf.shape[0] for i in range(traindf.shape[0])])

rfClf = AdaBoost(DTreeClassifier,int(argv[1]))
rfClf.fit(traindf,CLASS,{'depth':int(argv[2])},{'skip':['bruises?-no']})
rfClf.score(testdf)
