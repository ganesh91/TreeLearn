from sys import argv
from mllib.datastructures.DataFrame import dataframe
from mllib.utils.Utils import test_train_split
from mllib.trees.dtree import DTreeClassifier
from mllib.ensemble.RandomForestClassifier import RandomForestClassifier
import time

if __name__ == '__main__':
	traindf=dataframe().read_csv("agaricuslepiotatrain1.csv",delimiter=",")
	testdf=dataframe().read_csv("agaricuslepiotatest1.csv",delimiter=",")
	traindf.createColumn('_weight',[1 for i in range(traindf.shape[0])])
	rfClf = RandomForestClassifier(int(argv[1]),int(argv[2]),'actual',4)
	start=time.time()
	rfClf.fit(traindf,'bruises?-bruises',{},{'skip':['bruises?-no']})
	rfClf.score(testdf)
	print((time.time()-start)/60)
