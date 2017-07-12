import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from  sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
#######################################################################
#read_file and preparation 
train_v=pd.read_csv('../input/training_variants')
test_v=pd.read_csv('../input//test_variants')
train_t=pd.read_csv('../input/training_text',sep="\|\|",engine="python",skiprows=1,names=["ID","Text"])
test_t=pd.read_csv('../input//test_text',sep="\|\|",engine="python",skiprows=1,names=["ID","Text"])
train=pd.merge(train_v,train_t,how='left',on='ID')
train["TextLen"]=train["Text"].map(lambda x:str(len(str(x)))+str(len(str(x).split())))
train_Y=train['Class'].values-1
train_X=train.drop('Class',axis=1)
test=pd.merge(test_v,test_t,how='left',on='ID')
test["TextLen"]=test["Text"].map(lambda x:str(len(str(x)))+str(len(str(x).split())))
ID=test["ID"].values
train_X.head(5)
test.head(5)
#########################################################################################################
#data exploration
def print_numbers():
	print "number of Genes in training data: ",len(train_v.Gene.unique()), "; number of Genes in test file: ",len(test_v.Gene.unique())
	print "number of Variations in training data: ",len(train_v.Variation.unique()), "; number of Variations in test file: ",len(test_v.Variation.unique())
	print "number of Texts in training data: ",len(train_X.Text.unique()), "; number of Text in test file: ", len(test.Text.unique())
	print train.head(5)
	print test.head(5)
	

def plot_classes():
	plt.figure(figsize=(12,8))
	sns.countplot(x='Class',data=train)
	plt.ylabel('Count',fontsize=14)
	plt.xlabel('Class',fontsize=14)
	plt.show()	

def plot_items():
	items=["TextLen","Gene","Variation"]
	for item in items:
		gene_group=train.groupby(item)["ID"].count().reset_index()
		top5=gene_group.sort_values('ID',ascending=False)
		top5=top5[:5]
		#print gene_top5 
		print top5.head(5)
		sns.barplot(x=item,y='ID',data=top5)
	
	gene=train[train.Variation=="Truncating Mutations"]
	classes=gene.groupby("Class")["Class"].count()
	print classes.sort_values(ascending=False)[:5]

	plt.figure(figsize=(12,8))
	sns.countplot(x='Class',data=gene)
	plt.xlabel('Class')
	plt.show()

def data_explore():
	print_numbers()
	plot_classes()
	plot_items()


#data_explore()

##########################################################################################################
#modeling 

#interpreting the variation string 
def variation_filter(vari):
	
	def is_num(num):
		if not num:
			return False
		nums="0123456789"
		check=True
		for c in num:
			if c not in nums:
				check=False
		return check
		
	if (len(vari)>=3): 
		alphabeta="GAVLIFWYDHNEKQMRSTCP"
		aa=["Gly","Ala","Val","Leu","Ile","Phe","Trp","Tyr","Asp","His","Asn","Glu","Lys","Gln","Met","Arg","Ser","Thr","Cys","Pro"]
		head=vari[0]
		end=vari[-1]
		body=vari[1:-1]
		if (head in alphabeta) and (end in alphabeta) and is_num(body):
			return vari+" "+aa[alphabeta.find(head)]+" "+aa[alphabeta.find(end)]
		else:
			return vari
	else:
		return vari

train_X["Variation"]=train_X["Variation"].map(lambda x: variation_filter(x))
test["Variation"]=test["Variation"].map(lambda x: variation_filter(x))
print "Variation filtered"

def model_build():
	class col_select(BaseEstimator,TransformerMixin):
		def __init__(self,key):
			self.key=key
		def fit(self,x,y=None):
			return self
		def transform(self,x):
			return x[self.key].apply(str)

	text_pipeline=Pipeline([('col',col_select("Text")),
	                        ('vector',CountVectorizer(ngram_range=(1,2))),
		                    ('tfidf',TfidfTransformer(use_idf=True,smooth_idf=True)),
		                    ('tsvd',TruncatedSVD(n_components=60,n_iter=30,random_state=1))
		                     ])

	gene_pipeline=Pipeline([('col',col_select("Gene")),
		                    #('vector',CountVectorizer(analyzer=u'char',ngram_range=(1,5))),
		                    ('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(1,5),use_idf=True,smooth_idf=True)),
		                    ('tsvd',TruncatedSVD(n_components=60,n_iter=30,random_state=1))
	                         ])

	variation_pipeline=Pipeline([('col',col_select("Variation")),
		                    #('vector',CountVectorizer(analyzer=u'char',ngram_range=(1,5))),
		                    ('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(1,5),use_idf=True,smooth_idf=True)),
		                    ('tsvd',TruncatedSVD(n_components=60,n_iter=30,random_state=1))
	                         ])


	process=Pipeline([
		('union',FeatureUnion(
			n_jobs=-1,
			transformer_list=[("text",text_pipeline),("gene",gene_pipeline),("variation",variation_pipeline)],
			#transformer_weights={'text':1.1,"gene":0.9,'variation':0.9}
			)
			)])

	print "model step1 finished"


	process.fit(train_X)
	print "fitted"


	train_trans=process.transform(train_X)
	print "training data process:",train_trans.shape
	test_trans=process.transform(test)
	print "test data process:", test_trans.shape


	joblib.dump(process,'process1')
	joblib.dump(train_trans,"train_transform")
	print "train_trans saved"
	joblib.dump(test_trans,"test_transform")
	print "test_trans saved"
	return train_trans,test_trans

def model_load():
	train_trans=joblib.load("train_transform")
	test_trans=joblib.load("test_transform")
	return train_trans,test_trans


#choose model_build at first time or when you want to refine the model, otherwise choose model_load to save time on fitting and transform 
train_trans,test_trans=model_load()


#####################################################################################################################
#training


def xgboost_prediction():
	iter=3
	for i in range(iter):
		x1,x2,y1,y2=train_test_split(train_trans,train_Y,test_size=0.15,random_state=iter)
		validation=[(xgb.DMatrix(x1,y1),'train'),(xgb.DMatrix(x2,y2),'validation')]
		params={'max_depth':5,'eta':0.01,'silent':1,'num_class':9,'eval_metric':'mlogloss','objective':'multi:softprob','seed':1}
		num_round=1500
		model=xgb.train(params,xgb.DMatrix(x1,y1),num_round,validation,verbose_eval=5,early_stopping_rounds=30)
		if i==0:
			test_Y=model.predict(xgb.DMatrix(test_trans),ntree_limit=model.best_ntree_limit)/iter
		else:
			test_Y+=model.predict(xgb.DMatrix(test_trans),ntree_limit=model.best_ntree_limit)/iter
	submission=pd.DataFrame(test_Y,columns=["class"+str(i) for i in range(1,10)])
	submission["ID"]=ID
	submission.to_csv("submission.csv",index=False)







def prediction(k=0):
	models=[SVC(kernel='linear',probability=True),MLPClassifier(activation='relu',learning_rate='adaptive',warm_start=True)]
	names=['SVC','MLPC']
	test_Y=np.zeros((test.shape[0],9))
	model=models[k]
	model.fit(train_trans,train_Y)
	test_Y+=model.predict_proba(test_trans)
	submission=pd.DataFrame(test_Y,columns=["class"+str(i) for i in range(1,10)])
	submission["ID"]=ID
	name="submission"+names[k]+".csv"
	submission.to_csv(name,index=False)

prediction(1)
#xgboost_prediction()