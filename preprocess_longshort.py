import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F	   
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import pickle 
import time
import os
#这个 sliding_varlen 函数的目的可能是处理具有不同长度的时间序列数据，并计算每个批次数据的熵
def sliding_varlen(data,batch_size):#定义了一个名为 sliding_varlen 的函数，它接受两个参数：data 和 batch_size。函数内部定义了两个辅助函数：timedelta 和 get_entropy
	def timedelta(time1,time2):#嵌套函数，它接受两个参数：time1 和 time2。
		t1 = datetime.datetime.strptime(str(time1),'%a %b %d %H:%M:%S %z %Y')#使用 datetime.strptime 方法将字符串 time1 转换为 datetime 对象 t1，按照给定的格式 '%a %b %d %H:%M:%S %z %Y' 解析。
		t2 = datetime.datetime.strptime(str(time2),'%a %b %d %H:%M:%S %z %Y')
		delta = t1-t2#计算两个 datetime 对象 t1 和 t2 之间的时间差。
		time_delta = datetime.timedelta(days = delta.days,seconds = delta.seconds).total_seconds()#创建一个新的 timedelta 对象，包含 delta.days 天和 delta.seconds 秒，然后调用 total_seconds() 方法将其转换为总秒数。
		return time_delta/3600#将总秒数转换为小时数（除以3600）并返回。
	
	def get_entropy(x):#在 sliding_varlen 函数内部定义了一个名为 get_entropy 的嵌套函数，它接受一个参数：x。
		x_value_list = set([x[i] for i in range(x.shape[0])])#创建一个包含 x 数组中所有唯一值的集合 x_value_list。
		ent = 0.0#初始化熵 ent 为0.0。
		for x_value in x_value_list:#遍历集合 x_value_list 中的每个元素。
			p = float(x[x == x_value].shape[0]) / x.shape[0]#计算当前值 x_value 在数组 x 中出现的概率 p。
			logp = np.log2(p)#计算概率 p 的以2为底的对数 logp
			ent -= p * logp#更新熵 ent，减去当前值的概率乘以其对数。
		return ent#返回计算得到的熵值。

#################################################################################
	#这段代码的主要目的是对一个名为 data 的数据集进行处理，将其按照时间戳进行排序，并提取出时间相关的特征
	# 1、sort the raw data in chronological order
	timestamp = [] #初始化一个空列表 timestamp，用于存储时间戳。
	hour = [] #初始化一个空列表 hour，用于存储小时信息。
	day = [] #初始化一个空列表 day，用于存储一年中的第几天。
	week = [] #初始化一个空列表 week，用于存储一周中的第几天。
	hour_48 = [] #初始化一个空列表 hour_48，用于存储调整后的小时信息，用于区分周末和工作日。
	for i in range(len(data)): #遍历数据集 data 的每一行。
		times = data['time'].values[i] #从数据集中提取第 i 行的 'time' 列的值。
		timestamp.append(time.mktime(time.strptime(times, '%a %b %d %H:%M:%S %z %Y'))) #将时间字符串 times 转换为时间结构，然后转换为时间戳，并将其添加到 timestamp 列表中。
		t = datetime.datetime.strptime(times,'%a %b %d %H:%M:%S %z %Y') #将时间字符串 times 转换为 datetime 对象。
		#datetime.strptime 方法能够解析符合特定格式的日期和时间字符串，将其转换为 datetime 对象。
		year = int(t.strftime('%Y')) #从 datetime 对象中提取年份，并转换为整数。
		day_i = int(t.strftime('%j')) #从 datetime 对象中提取一年中的第几天，并转换为整数。
		week_i = int(t.strftime('%w')) #
		hour_i = int(t.strftime('%H'))
		#初始化 hour_i_48 为当前小时 hour_i
		hour_i_48 = hour_i
		if week_i == 0 or week_i == 6: #检查当前日期是否为周六（0）或周日（6）。
			hour_i_48 = hour_i + 24 #如果是周末，则将小时数增加24，以区分周末和工作日。

		if year == 2013: #如果是2013年，则将 day_i 增加366，因为2013年不是闰年，这里可能是为了某种特定的处理。
			day_i = day_i + 366
		day.append(day_i) #将计算后的 day_i 添加到 day 列表中。
		hour.append(hour_i) 
		hour_48.append(int(hour_i_48))
		week.append(week_i)

	data['timestamp'] = timestamp #将 timestamp 列表添加到数据集 data 中，作为新的列。
	data['hour'] = hour
	data['day'] = day
	data['week'] = week
	data['hour_48'] = hour_48

	data.sort_values(by = 'timestamp',inplace=True,ascending = True) #按照 timestamp 列的值对数据集 data 进行排序，inplace=True 表示在原数据集上进行排序，ascending=True 表示按升序排序。

#################################################################################
	# 2、filter users and POIs 
	#这段代码的主要目的是对数据集 data 进行过滤和处理，以筛选出活跃的用户和兴趣点（POIs），并最终生成一个包含POI类别信息的张量 cat_candidate。
	'''
	thr_venue = 1  # 设置阈值。通过设置阈值，可以专注于那些足够活跃的用户和POIs。例如，thr_venue 为1意味着只考虑至少被访问两次的POIs。
	thr_user = 20  #业务需求可能需要关注那些达到一定活跃度的用户和POIs。例如，一个商家可能只对那些至少有20个用户访问过的POIs感兴趣。
	user_venue = data.loc[:,['userid','venueid']]  # 提取用户ID和POI ID
	#user_venue = user_venue.drop_duplicates()  
	
	venue_count = user_venue['venueid'].value_counts()
	venue = venue_count[venue_count.values>thr_venue]
	venue_index =  venue.index
	data = data[data['venueid'].isin(venue_index)]
	user_venue = user_venue[user_venue['venueid'].isin(venue_index)]
	del venue_count,venue,venue_index
	
	#user_venue = user_venue.drop_duplicates()
	user_count = user_venue['userid'].value_counts()
	user = user_count[user_count.values>thr_user]
	user_index = user.index
	data = data[data['userid'].isin(user_index)]
	user_venue = user_venue[user_venue['userid'].isin(user_index)]
	del user_count,user,user_index
	
	user_venue = user_venue.drop_duplicates()
	user_count = user_venue['userid'].value_counts()
	user = user_count[user_count.values>1]
	user_index = user.index
	data = data[data['userid'].isin(user_index)]
	del user_count,user,user_index
	
	'''
	data['userid'] = data['userid'].rank(method='dense').values
	data['userid'] = data['userid'].astype(int)
	data['venueid'] =data['venueid'].rank(method='dense').values
	data['userid'] = data['userid'].astype(int)
	for venueid,group in data.groupby('venueid'):
		indexs = group.index
		if len(set(group['catid'].values))>1:
			for i in range(len(group)):
				data.loc[indexs[i],'catid'] = group.loc[indexs[0]]['catid']
	
	data = data.drop_duplicates()
	data['catid'] =data['catid'].rank(method='dense').values
	
#################################################################################
	poi_cat = data[['venueid','catid']]
	poi_cat = poi_cat.drop_duplicates()
	poi_cat = poi_cat.sort_values(by = 'venueid')
	cat_candidate = torch.Tensor(poi_cat['catid'].values)

	with open('cat_candidate.pk','wb') as f:
		pickle.dump(cat_candidate,f)

	# 3、split data into train set and test set.
	#    extract features of each session for classification

	vocab_size_poi = int(max(data['venueid'].values))
	vocab_size_cat = int(max(data['catid'].values))
	vocab_size_user = int(max(data['userid'].values))

	print('vocab_size_poi: ',vocab_size_poi)
	print('vocab_size_cat: ',vocab_size_cat)
	print('vocab_size_user: ',vocab_size_user)

	train_x  = []
	train_x_cat  = []
	train_y = []
	train_hour = []
	train_userid = []
	train_indexs = []

# the hour and week to be predicted
	train_hour_pre = []
	train_week_pre = []


	test_x  = []
	test_x_cat  = []
	test_y = []
	test_hour = []
	test_userid = []
	test_indexs = []

# the hour and week to be predicted
	test_hour_pre = []
	test_week_pre = []


	long_term = {}

	long_term_feature = []

	data_train = {}
	train_idx = {}
	data_test = {}
	test_idx = {}

	data_train['datainfo'] = {'size_poi':vocab_size_poi+1,'size_cat':vocab_size_cat+1,'size_user':vocab_size_user+1} 
	
	len_session = 20
	user_lastid = {}
#################################################################################
	# split data

	for uid, group in data.groupby('userid'):
		data_train[uid] = {}
		data_test[uid] = {}
		user_lastid[uid] = []
		inds_u = group.index.values
		split_ind = int(np.floor(0.8*len(inds_u)))
		train_inds = inds_u[:split_ind]
		test_inds = inds_u[split_ind:]

	#get the features of POIs for user uid
		#long_term_feature.append(get_features(group.loc[train_inds]))

		long_term[uid] = {}
		'''
		long_term[uid]['loc'] = []
		long_term[uid]['hour'] = []
		long_term[uid]['week'] = []
		long_term[uid]['category'] = []
	
		lt_data = group.loc[train_inds]
		long_term[uid]['loc'].append(lt_data['venueid'].values)
		long_term[uid]['hour'].append(lt_data['hour'].values)
		long_term[uid]['week'].append(lt_data['week'].values)
		long_term[uid]['category'].append(lt_data['catid'].values)
		'''
		lt_data = group.loc[train_inds]
		long_term[uid]['loc'] = torch.LongTensor(lt_data['venueid'].values).cuda()
		long_term[uid]['hour'] = torch.LongTensor(lt_data['hour'].values).cuda()
		long_term[uid]['week'] = torch.LongTensor(lt_data['week'].values).cuda()
		long_term[uid]['category'] = torch.LongTensor(lt_data['catid'].values).cuda()
	#split the long sessions to some short ones with len_session = 20

		train_inds_split = []
		num_session_train =int(len(train_inds)//(len_session))
		for i in range(num_session_train):
			train_inds_split.append(train_inds[i*len_session:(i+1)*len_session])
		if num_session_train<len(train_inds)/len_session:
			train_inds_split.append(train_inds[-len_session:])
		
		train_id = list(range(len(train_inds_split))) 

		test_inds_split = []
		num_session_test = int(len(test_inds)//(len_session))
		for i in range(num_session_test):
			test_inds_split.append(test_inds[i*len_session:(i+1)*len_session])
		if num_session_test<len(test_inds)/len_session:
			test_inds_split.append(test_inds[-len_session:])
		
		test_id = list(range(len(test_inds_split)+len(train_inds_split)))[-len(test_inds_split):]

		train_idx[uid] = train_id[1:]
		test_idx[uid] = test_id

		for ind in train_id:

		#generate data for comparative methods such as deepmove

			if ind == 0:
				continue

			data_train[uid][ind] = {}
			history_ind =[]
			for i in range(ind):
				history_ind.extend(train_inds_split[i])
			whole_ind = []
			whole_ind.extend(history_ind)
			whole_ind.extend(train_inds_split[ind])

			whole_data = group.loc[whole_ind]

			loc = whole_data['venueid'].values[:-1]
			tim = whole_data['hour'].values[:-1]
			target = group.loc[train_inds_split[ind][1:]]['venueid'].values

			#loc = group_i['venueid'].values[:-1]
			#tim = get_day(group_i['time'].values)[1][:-1]
			#target = group_i['venueid'].values[-10:]

			data_train[uid][ind]['loc'] = torch.LongTensor(loc).unsqueeze(1)
			data_train[uid][ind]['tim'] = torch.LongTensor(tim).unsqueeze(1)
			data_train[uid][ind]['target'] = torch.LongTensor(target)

			user_lastid[uid].append(loc[-1])

			group_i = group.loc[train_inds_split[ind]]
		
			#generate data for SHAN
			current_loc = group_i['venueid'].values
			data_train[uid][ind]['current_loc'] = torch.LongTensor(current_loc).unsqueeze(1)
			#group_i = whole_data
		#generate data for my methods. X,Y,time,userid
			
			current_cat = group_i['catid'].values
			train_x.append(current_loc[:-1])
			train_x_cat.append(current_cat[:-1])
			train_y.append(current_loc[1:])
			#train_hour.append(group_i['hour_48'].values[:-1])
			train_hour.append(group_i['hour'].values[:-1])
			train_userid.append(uid)
			
			#train_hour_pre.append(group_i['hour'].values[-1])
			#train_week_pre.append(group_i['week'].values[-1])

			train_hour_pre.append(group_i['hour'].values[1:])
			train_week_pre.append(group_i['week'].values[1:])

			train_indexs.append(group_i.index.values)

		for ind in test_id:

			data_test[uid][ind] = {}
			history_ind =[]
			for i in range(len(train_inds_split)):
				history_ind.extend(train_inds_split[i])
			whole_ind = []
			whole_ind.extend(history_ind)
			whole_ind.extend(test_inds_split[ind-len(train_inds_split)])

			whole_data = group.loc[whole_ind]

			loc = whole_data['venueid'].values[:-1]
			tim = whole_data['hour'].values[:-1]
			target = group.loc[test_inds_split[ind-len(train_inds_split)][1:]]['venueid'].values

			#loc = group_i['venueid'].values[:-1]
			#tim = get_day(group_i['time'].values)[1][:-1]
			#target = group_i['venueid'].values[-10:]

			data_test[uid][ind]['loc'] = torch.LongTensor(loc).unsqueeze(1)
			data_test[uid][ind]['tim'] = torch.LongTensor(tim).unsqueeze(1)
			data_test[uid][ind]['target'] = torch.LongTensor(target)

			user_lastid[uid].append(loc[-1])

			#group_i = whole_data

			group_i = group.loc[test_inds_split[ind-len(train_inds_split)]]

			current_loc = group_i['venueid'].values
			data_test[uid][ind]['current_loc'] = torch.LongTensor(current_loc).unsqueeze(1)

			current_cat = group_i['catid'].values
			test_x_cat.append(current_cat[:-1])
			test_x.append(current_loc[:-1])
			test_y.append(current_loc[1:])
			#test_hour.append(group_i['hour_48'].values[:-1])
			test_hour.append(group_i['hour'].values[:-1])
			test_userid.append(uid)

			#test_hour_pre.append(group_i['hour'].values[-1])
			#test_week_pre.append(group_i['week'].values[-1])

			test_hour_pre.append(group_i['hour'].values[1:])
			test_week_pre.append(group_i['week'].values[1:])

			test_indexs.append(group_i.index.values)


	with open('data_train.pk','wb') as f:
		pickle.dump(data_train,f)

	with open('data_test.pk','wb') as f:
		pickle.dump(data_test,f)

	with open('train_idx.pk','wb') as f:
		pickle.dump(train_idx,f)

	with open('test_idx.pk','wb') as f:
		pickle.dump(test_idx,f)

	print('user_num: ',len(data_train.keys()))
	#minMax = MinMaxScaler()
	#long_term_feature = minMax.fit_transform(np.array(long_term_feature))

	with open('long_term.pk','wb') as f:
		pickle.dump(long_term,f)

	#with open('long_term_feature.pk','wb') as f:
	#	pickle.dump(long_term_feature,f)

	def dataloader(X,X_cat,Y,hour,userid,hour_pre,week_pre):
		
		torch_dataset = Data.TensorDataset(torch.LongTensor(X),torch.LongTensor(X_cat),torch.LongTensor(Y),torch.LongTensor(hour),torch.LongTensor(userid),torch.LongTensor(hour_pre),torch.LongTensor(week_pre))
		loader = Data.DataLoader(
			dataset = torch_dataset,  
			batch_size = batch_size,  
			shuffle = True,
			num_workers = 0,
		)
		return loader

	loader_train = dataloader(train_x,train_x_cat,train_y,train_hour,train_userid,train_hour_pre,train_week_pre)
	loader_test = dataloader(test_x,test_x_cat,test_y,test_hour,test_userid,test_hour_pre,test_week_pre)

	pre_data = {}
	pre_data['size'] = [vocab_size_poi,vocab_size_cat,vocab_size_user,len(train_x),len(test_x)]
	pre_data['loader_train'] = loader_train
	pre_data['loader_test'] = loader_test

	with open('pre_data.txt','wb') as f:
		pickle.dump(pre_data,f)
	return pre_data
