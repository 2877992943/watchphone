#!/usr/bin/env python
# encoding=utf-8

"""


"""

 

#dataPath='/home/yr/watchphone/data1019/' 

para_pack=0#1with liangbin riding into trainset 
import numpy as np
import pylab as plt
import cPickle,math,random,theano
import theano.tensor as T
import lasagne
from leancloud import Object
from leancloud import Query
import leancloud
import time,os,operator


deviceId={'huawei':'ffffffff-c7a8-3cd1-ffff-ffffea16e571', 
	'xiaomi':'ffffffff-c7a7-c7d4-0000-000031a92fbe', 
	'mine':'ffffffff-c95d-196f-0000-00007e282437',
	'chenyu':'ffffffff-c43f-692d-ffff-fffff4d57110',
	'wangx':'ffffffff-c28a-1bf0-0000-00005e774605',
	'zhangx':'ffffffff-f259-8a54-0000-0000300c3e8c',
	'liyt':'ffffffff-c7a8-3c91-ffff-ffffa2dc21fc',
	'donggy':'ffffffff-c95e-eae5-ffff-ffffce758366',
	'hangwz':'ffffffff-c43f-77d4-0000-00001b78f081',
	'zishuai':'ffffffff-9475-8052-ffff-ffffaa303f0b'
	}
sensor_list=["magneticSensor","accelerometer","proximity",]  
#proximity 60second interval 2value   30sec capture 1value


watch_nov_zs=[[5,12,10,5,12,12],[5,12,29,5,12,37],[5,15,11,5,15,15]]
notwatch_nov_zs=[[5,15,33,5,15,39]]
watch_nov_jw=[[4,19,55,4,20,5]]
nov_vivo=[[5,15,57,5,16,1],[5,16,6,5,16,8],[7,13,7,7,13,12],[7,17,4,7,17,15]]
c_list=['watch','notwat']
inst_id_zs='mTl3M9NJ0qjkKOKFX378VWa37IFYKBe5'
inst_id_hjw='rumE011he7vtJxkINHkHQTdhkoBjJMcr'
inst_id_vivo='czte5wJCwAFRSUJWsC5ybyaezAhDnOm1'
###########query label.data
device=deviceId['chenyu'] 
period=nov_vivo[-1]
#duration=watch_nov5
###########query log.tracer
inst_id=inst_id_zs
#period=watch_nov_zs[-1]
###########combine train pack
class_type=c_list[0]#watch 0 | notwatch1

##########
smooth_window=10
###########3
num_stump=31
max_Dep=100#  usually 7,8 10stop, 200maxdepth not useful
each_step_gap=5#20 50     when choose best [fea_ind,fea_value]
dim_percentage=1#when random sample,random dim,how many feature 
best_tree_accuracy=0.97#ensemble with those single trees with what kind of accuracy
#############33

def generate_stamp(period):
	#[8,28,8,33]->[(2015, 10, 20, 22, 30, 0, 0, 0, 0),(2015, 10, 20, 22, 48, 0, 0, 0, 0)]->stamp
	dur= [(2015, 11, period[0], period[1], period[2], 0, 0, 0, 0),\
		(2015, 11, period[3], period[4], period[5], 0, 0, 0, 0)]
	stamp_range0=[time2stamp(dur[0]),time2stamp(dur[1])]
	stamp_range=[t*1000 for t in stamp_range0]
	return stamp_range 


def connect_db_log():##sensor.log.tracer   not label.sensor, 
	import leancloud,cPickle
	appid = "9ra69chz8rbbl77mlplnl4l2pxyaclm612khhytztl8b1f9o"
	   
	appkey = "1zohz2ihxp9dhqamhfpeaer8nh1ewqd9uephe9ztvkka544b"
	#appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	#appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)

def time2stamp(t):
	#t = (2015, 9, 28, 12, 36, 38, 0, 0, 0)
	stamp = int(time.mktime( t )) ;
	return stamp
def connect_db():#label.data
	import leancloud,cPickle
	appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)

 
 
def get_content_fromLabel(results):#result is from find() 
 	 
	obs={}; 
	r=results
	for i in range(1):
		#print type(r.get("events")) 
		if len(r.get("events"))>=1:
			 
			print r.get("motion"),r.get("events").__len__()
			ll=r.get("events") #ll=[ {},{}...]
			for dic in ll[:]:#dic={timestamp:xxxx,value:[1,2,3]...}
			
			#print dic["timestamp"],' ',dic["values"][0],' ',dic["values"][1],' ',dic["values"][2]
				if dic["timestamp"] not in obs.keys():
					obs[ dic["timestamp"] ]=[r.get("motion"),\
					dic["values"][0],dic["values"][1],dic["values"][2]  ]
				###data form: {timestamp:[obs],...}  [obs]=[motion,x,y,z]
		 
	###########################
	 
 
	return obs 

#####################
def get_content_fromLog(results):#result is <>
 	 
	obs={}; 
	r=results
	
	ll=r.get("value") #ll=[ {},{}...]
	for dic in ll["events"][:]:#dic={timestamp:xxxx,value:[1,2,3]...}
			
		#print dic["timestamp"],' ',dic["values"][0],' ',dic["values"][1],' ',dic["values"][2]
		if dic["timestamp"] not in obs.keys():
			obs[ dic["timestamp"] ]=[class_type,\
			dic["values"][0],dic["values"][1],dic["values"][2]  ]
			###data form: {timestamp:[obs],...}  [obs]=[motion,x,y,z]
		 
	###########################
	print 'final',obs.__len__()
	###################3
	return obs 
	 


def get_all(query,skip,result):
	limit=500
	query.limit(limit)
	query.skip(skip)
	found=query.find()
	if found and len(found)>0:
		result.extend(found)
		print 'av_utils get all,now result len:',len(result),'skip',skip
		return get_all(query,skip+limit,result)
	else:
		return result
	


def save2pickle(c,name):
    write_file=open(dataPath+str(name),'wb')
    cPickle.dump(c,write_file,-1)#[ (timestamp,[motion,x,y,z]),...]
    write_file.close()
 
def load_pickle(path_i):
    f=open(path_i,'rb')
    data=cPickle.load(f)#[ [time,[xyz],y] ,[],[]...]
    f.close()
    #print data.__len__(),data[0]
    return data	







def fea4(obs):#[50,]obs  fea21->fea4
	#4
	mean=np.mean(obs);std=np.std(obs)
	min_i=np.min(obs);max_i=np.max(obs)
	f=np.array([mean,std,min_i,max_i])
	dim=obs.shape[0]
	#percentile 5
	percentile=[10/100.*dim,25/100.*dim,50/100.*dim,75/100.*dim,90/100.*dim];#print percentile
	perc=[int(i) for i in percentile];#print perc
	obs_sort=np.sort(obs)#[50,]
	perc_i=obs_sort[percentile];#print perc_i#[5,]
	#sum, square-sum 12
	position=[5,10,25,75,90,95]
	pos=[int(i/100.*dim) for i in position];#print pos
	sum_i=[np.sum(obs_sort[:i]) for i in pos]
	sqrt_sum_i=[np.sqrt(np.dot(obs_sort[:i],obs_sort[:i])) for i in pos]
	#
	fea_i=np.concatenate((f,perc_i,sum_i,sqrt_sum_i),axis=0);#print fea_i.shape
	return fea_i[:4]#[21,]



 







def classify(inputTree,testVec):
	firstStr = inputTree.keys()[0]#[dim,value]
	dim1,v1=firstStr
	secondDict = inputTree[firstStr]
   
     
	if testVec[dim1] <=v1:#go left
        	if type(secondDict['left']).__name__ == 'dict':
			
                	classLabel = classify(secondDict['left'], testVec)
            	else: 
			classLabel = secondDict['left']
			 
	else:#go right
		if type(secondDict['right']).__name__ == 'dict':
                	classLabel = classify(secondDict['right'], testVec)
            	else: 
			classLabel = secondDict['right']
			 
				
	return classLabel



 


###############################
def db_label_data(period):
	#########################3
	#db label.data
	#########################3
	####init
	connect_db()
	stamp_range=generate_stamp(period) 
	#####get all
	UserSensor = Object.extend('UserSensor')
	query_acc = Query(UserSensor) 
	

	#stamp_range0=[time2stamp(duration[0]),time2stamp(duration[1])]
	#stamp_range=[t*1000 for t in stamp_range0]
	#query.not_equal_to("deviceId",None).not_equal_to("events",None).equal_to("motion",class_type)
	query_acc.equal_to("deviceId",device).not_equal_to("events",None).\
		equal_to("sensorType",'accelerometer').\
		less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
		
		#equal_to("motion",class_type).\
	 
	print 'count acc',query_acc.count() 
	acc_list=get_all(query_acc,0,[]);print 'acc',len(acc_list)#,all_list[0]#instance object
	 
	######acc  
	all_acc_list=[];i=0
	for obj in acc_list:
		obs=get_content_fromLabel(obj)#data form: {timestamp:[o],...}  [o]=[motion,x,y,z]
		all_acc_list.append(obs)#[{},...]
		i+=obs.__len__()
	print 'total',i,all_acc_list.__len__()
	 
	#####
	###############combine{} sort by timestamp
	data_dic={};acc_list=all_acc_list
    	for dic in acc_list:
		for k,v in dic.items():
			if k not in data_dic:
				data_dic[k]=v
	##sort acc
	ll=sorted(data_dic.items(),key=lambda f:f[0],reverse=False)
	# # DATA FORMATE  {timestamp:[motion x y z],...}->[ (timestamp,[motion,x,y,z]),...]
	xyz=np.array([obs[1][1:] for obs in ll]);print 'xyz',xyz.shape#[ [xyz],[]...] shape[n,3]
	#save2pickle(xyz,'xyz-watchphone-nov-'+class_type)
	return xyz



def main(xyz):
	#########################
	#some err
	##############
	xyz_acc=xyz
	mod_acc_mean=np.mean( np.sqrt( (xyz_acc*xyz_acc).sum(axis=1) ) )
	assert isinstance(xyz_acc,np.ndarray) 
	assert xyz_acc.shape[0]>=10 
	assert xyz_acc.shape[1]==3
	assert mod_acc_mean>=5 and mod_acc_mean<=50
	##########################################
	#fea visual  filter ->valid data
	#################################################
	#########load pickle
	#xyz=load_pickle(dataPath+'xyz-watchphone-nov-'+class_type)
	
	
	######clean data
	#xyz=xyz[:4087,:]
	  
	xyz_abs=np.abs(xyz) 
	
     
	#########visual
	x_axis=xyz.shape[0]#500 200 1000
	y_axis=30
    	ind=np.arange(xyz.shape[0])
	 
    	plt.figure()
	plt.subplot(211);plt.title('xyz')
    	plt.plot(ind,xyz[:,0],'r-',ind,xyz[:,1],'y.',ind,xyz[:,2],'b--');#plt.xlim(0,9000);#plt.ylim(0,y_axis)
	 
	#plt.show() 
 
	###########################
	#generate obs x y [n,3]->[n_obs,10,3] ->[n_obs,4x3]
	############################
	fea=xyz
	kernel_sz=10.;stride=kernel_sz;
    	obs_list=[]; 
    	num=int( (xyz.shape[0]-kernel_sz)/stride ) +1
    	for i in range(num)[:]: #[0,...100] total 101 
        	obs=fea[i*stride:i*stride+kernel_sz,:]#[10,3]
		if obs.shape[0]==kernel_sz:
			v=np.array([fea4(obs[:,i]) for i in range(obs.shape[1])]).flatten()
			obs_list.append(v)#[10,3]->[3x4,]
	x_arr=np.array(obs_list);print 'x',x_arr.shape#[n-obs,12]
	 
	 
	#####
	#save2pickle([x_arr,y_arr],'xy-'+class_type) #[n,12][n,]	
	 
	xy=np.concatenate((x_arr,np.zeros((x_arr.shape[0],1)) ),axis=1);print xy.shape #[n,13]
	
	#
	#dataSet=[list(xy[i,:]) for i in range(xy.shape[0]) ]  #nx13 list
	#dataSet=np.array([xy[i,:] for i in range(xy.shape[0]) ] )  #nx13  array
	#
	dataSet=xy
	#load model trees
 	many_stumps=load_pickle(dataPath+'rf-para-watchphone')
 	


	#test 
	X_test=dataSet#[n,13]
	#####ensemble test    [  [],[]...]  []=[tree,dim_list,accuracy]

	n_val=X_test.shape[0];dim_val=X_test.shape[1]-1#[n,13] 12+1
	f_label_mat=np.zeros((n_val,many_stumps.__len__())) #[n,10stump]
	
	for ind in range(many_stumps.__len__()):#[3,5,8,11..]
		dim_sample=many_stumps[ind][1]
		tree=many_stumps[ind][0]
		for obs in range(n_val):
			pred=classify(tree,X_test[obs,:])#13=12+1
			f_label_mat[obs,ind]=pred


	##
	#f_label majority vote
	 
	maj_vote=np.zeros((n_val,))#[n,]
	for i in range(f_label_mat.shape[0]): #[n,10stump]
		vote1=f_label_mat[i,:].sum()
		vote0=(1-f_label_mat[i,:]).sum()
		maj_vote[i]=[1 if vote1>vote0 else 0][0]
	##calculate accuracy
	#y_true=X_test[:,-1]
	#accuracy=T.mean(T.eq(maj_vote,y_true)).eval();print 'accur dt ensemble',accuracy

	plt.subplot(2,1,2);plt.title('predict watch or not watch over time')
	plt.plot(np.where(maj_vote==0)[0],np.zeros((np.where(maj_vote==0)[0].shape[0],)),'bo',label='watch');
	plt.plot(np.where(maj_vote==1)[0],np.ones((np.where(maj_vote==1)[0].shape[0],)),'y^',label='notwatch');
	plt.legend()
	plt.xlabel('time');plt.ylabel('predicted class')
	#plt.show()
	#class_dic={'watch':0,'notwatch':1}
	class_dic={0:'watch',1:'notwatch'}
	label_list=[class_dic[i] for i in maj_vote]
	print 'predict',label_list
	return label_list# [string,...]predict label

		

############3
if __name__=="__main__":

	#put data and parameter in the path below
	dataPath='/home/yr/watchphone/data1019/' 
	##read xyz from label.data database
	#xyz=db_label_data(period) 
	xyz=load_pickle(dataPath+'xyz-watchphone-nov')
 	#predict and visual
	predicted_label_list=main(xyz)
	 
	plt.show()
		 
	
	
	

	 
	
	 
	

	
	 
	 

	 

 

	 
	
	 

	
    
 
	
		
	
   		 



