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
	#user_venue = user_venue.drop_duplicates()  #这行代码被注释掉了，如果启用，它会删除user_venue中的重复行。
	
	venue_count = user_venue['venueid'].value_counts() #计算每个POI的访问次数。
	venue = venue_count[venue_count.values>thr_venue] #筛选出访问次数超过阈值thr_venue的POIs。
	venue_index =  venue.index #：获取筛选出的POIs的索引。
	data = data[data['venueid'].isin(venue_index)] #在原始数据集中只保留那些POI ID在venue_index中的行。
	user_venue = user_venue[user_venue['venueid'].isin(venue_index)] #在用户-POI数据集中只保留那些POI ID在venue_index中的行。
	del venue_count,venue,venue_index #删除不再需要的变量以释放内存。
	
	#user_venue = user_venue.drop_duplicates() 这行代码被注释掉了，如果启用，它会删除user_venue中的重复行。
	user_count = user_venue['userid'].value_counts() 计算每个用户的访问次数
	user = user_count[user_count.values>thr_user] 筛选出访问次数超过阈值thr_user的用户。
	user_index = user.index 获取筛选出的用户索引。
	data = data[data['userid'].isin(user_index)] 在原始数据集中只保留那些用户ID在user_index中的行。
	user_venue = user_venue[user_venue['userid'].isin(user_index)] 在用户-POI数据集中只保留那些用户ID在user_index中的行。
	del user_count,user,user_index  删除不再需要的变量以释放内存。
	
	user_venue = user_venue.drop_duplicates() 删除user_venue中的重复行。
	user_count = user_venue['userid'].value_counts() 再次计算每个用户的访问次数。
	user = user_count[user_count.values>1] 筛选出访问次数超过1次的用户。
	user_index = user.index 获取筛选出的用户索引。
	data = data[data['userid'].isin(user_index)]  在原始数据集中只保留那些用户ID在user_index中的行。
	del user_count,user,user_index  #删除不再需要的变量以释放内存。
	
	'''
	data['userid'] = data['userid'].rank(method='dense').values 为userid字段中的每个值分配一个排名，使用dense方法，意味着排名之间的差距总是1，即使有相同的值
	data['userid'] = data['userid'].astype(int) 将userid字段的数据类型转换为整数。
	data['venueid'] =data['venueid'].rank(method='dense').values 为venueid字段中的每个值分配一个排名。
	data['userid'] = data['userid'].astype(int) 这里应该是一个错误，因为上一行代码已经将userid转换为整数了，这行代码实际上将venueid的排名结果赋值给了userid，并将数据类型转换为整数。
	for venueid,group in data.groupby('venueid'): 按venueid对数据进行分组。
		indexs = group.index 获取当前分组的索引。
		if len(set(group['catid'].values))>1:检查当前分组中catid的唯一值数量是否大于1。
			for i in range(len(group)): 遍历当前分组的每一行。
				data.loc[indexs[i],'catid'] = group.loc[indexs[0]]['catid'] 如果当前分组的catid值不唯一，那么将当前行的catid设置为该分组第一行的catid值。
	
	data = data.drop_duplicates() 删除数据集中的重复行。
	data['catid'] =data['catid'].rank(method='dense').values 为catid字段中的每个值分配一个排名。
	
#################################################################################
	poi_cat = data[['venueid','catid']] 从数据集中提取venueid和catid两列，创建一个新的DataFrame。
	poi_cat = poi_cat.drop_duplicates() 删除poi_cat中的重复行。
	poi_cat = poi_cat.sort_values(by = 'venueid') 按照venueid对poi_cat进行排序。
	cat_candidate = torch.Tensor(poi_cat['catid'].values) 将poi_cat中的catid列的值转换为PyTorch的Tensor对象。

	with open('cat_candidate.pk','wb') as f:  使用pickle模块将cat_candidate Tensor对象序列化并保存到前面打开的文件中。
		pickle.dump(cat_candidate,f)

	# 3、split data into train set and test set.
	#    extract features of each session for classification

	vocab_size_poi = int(max(data['venueid'].values)) 计算POI（兴趣点）的最大ID值，用于确定POI词汇表的大小。
	vocab_size_cat = int(max(data['catid'].values))  计算类别（catid）的最大ID值，用于确定类别词汇表的大小。
	vocab_size_user = int(max(data['userid'].values)) 计算用户（userid）的最大ID值，用于确定用户词汇表的大小。
          #打印POI、类别和用户的词汇表大小。
	print('vocab_size_poi: ',vocab_size_poi)
	print('vocab_size_cat: ',vocab_size_cat)
	print('vocab_size_user: ',vocab_size_user)
          #初始化多个空列表，用于存储训练集和测试集的特征、标签、时间、用户ID和索引。
	train_x  = []  初始化一个空列表 train_x，可能用于存储训练集中的特征数据。
	train_x_cat  = []  初始化一个空列表 train_x_cat，可能用于存储训练集中的分类特征数据。
	train_y = []  初始化一个空列表 train_y，可能用于存储训练集中的目标变量或标签。
	train_hour = []  初始化一个空列表 train_hour，可能用于存储训练集中的时间数据（小时）。
	train_userid = []  初始化一个空列表 train_userid，可能用于存储训练集中的用户ID。
	train_indexs = []   初始化一个空列表 train_indexs，可能用于存储训练集中的索引

# the hour and week to be predicted
	train_hour_pre = []  初始化一个空列表 train_hour_pre，可能用于存储预测时使用的训练集小时数据。
	train_week_pre = []   初始化一个空列表 train_week_pre，可能用于存储预测时使用的训练集周数据。


	test_x  = []
	test_x_cat  = []
	test_y = []
	test_hour = []
	test_userid = []
	test_indexs = []

# the hour and week to be predicted
	test_hour_pre = []
	test_week_pre = []


	long_term = {}       初始化一个空字典 long_term，可能用于存储长期特征。

	long_term_feature = []   初始化一个空列表 long_term_feature，可能用于存储长期特征。
	data_train = {}  初始化一个空字典 data_train，可能用于存储训练集的相关信息。
	train_idx = {}  
	data_test = {}
	test_idx = {} 初始化一个空字典 test_idx，可能用于存储测试集的索引信息。

	data_train['datainfo'] = {'size_poi':vocab_size_poi+1,'size_cat':vocab_size_cat+1,'size_user':vocab_size_user+1} 在 data_train 字典中添加一个键 datainfo，其值为另一个字典，包含三个键：size_poi、size_cat 和 size_user，分别表示兴趣点（POI）的词汇大小加1、分类特征的词汇大小加1和用户ID的词汇大小加1。
	
	len_session = 20  定义一个变量 len_session 并赋值为20，可能表示会话的长度或某个特定的序列长度。
	user_lastid = {}  初始化一个空字典 user_lastid，可能用于存储每个用户的最后一个ID。
#################################################################################
	# split data
        

	for uid, group in data.groupby('userid'): 遍历数据集中的每个用户（userid）。
		data_train[uid] = {} 初始化一个字典来存储训练数据。
		data_test[uid] = {} 初始化一个字典来存储测试数据。
		user_lastid[uid] = [] 初始化一个列表来存储用户最后访问的ID。
		inds_u = group.index.values 获取当前用户的所有数据索引。
		split_ind = int(np.floor(0.8*len(inds_u))) 计算分割点，将80%的数据用作训练集。
		train_inds = inds_u[:split_ind] 分割出训练集的索引。
		test_inds = inds_u[split_ind:]  分割出测试集的索引。

	#get the features of POIs for user uid
		#long_term_feature.append(get_features(group.loc[train_inds]))

		long_term[uid] = {}  初始化一个字典来存储长期特征。
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
		lt_data = group.loc[train_inds]  选取训练集的数据。
		long_term[uid]['loc'] = torch.LongTensor(lt_data['venueid'].values).cuda() 提取训练集中的位置特征，并转换为张量，然后移动到GPU。
		long_term[uid]['hour'] = torch.LongTensor(lt_data['hour'].values).cuda() 提取训练集中的小时特征，并转换为张量，然后移动到GPU。
		long_term[uid]['week'] = torch.LongTensor(lt_data['week'].values).cuda() 提取训练集中的周特征，并转换为张量，然后移动到GPU。
		long_term[uid]['category'] = torch.LongTensor(lt_data['catid'].values).cuda() 提取训练集中的类别特征，并转换为张量，然后移动到GPU。
	#split the long sessions to some short ones with len_session = 20 
目的是将用户的长期会话（long sessions）分割成多个短期会话（short sessions），每个短期会话的长度由变量 len_session 定义，这里设置为20。这样做的原因可能是为了模拟用户在短时间内的行为模式，这在某些推荐系统或行为预测模型中是很常见的。
		train_inds_split = [] 初始化一个列表来存储分割后的训练集索引。
		num_session_train =int(len(train_inds)//(len_session)) 计算可以分割成多少个长度为len_session的训练集。
		for i in range(num_session_train): 遍历每个分割的测试集
			train_inds_split.append(train_inds[i*len_session:(i+1)*len_session]) 将分割后的训练集索引添加到列表中
		if num_session_train<len(train_inds)/len_session: 如果训练集的最后一个分割不完整，则添加剩余的索引。
			train_inds_split.append(train_inds[-len_session:]) 
		
		train_id = list(range(len(train_inds_split)))  创建训练集的ID列表

		test_inds_split = [] 初始化一个列表来存储分割后的测试集索引。
		num_session_test = int(len(test_inds)//(len_session)) 计算可以分割成多少个长度为len_session的测试集。
		for i in range(num_session_test): 遍历每个分割的测试集。
			test_inds_split.append(test_inds[i*len_session:(i+1)*len_session]) 将分割后的测试集索引添加到列表中。
		if num_session_test<len(test_inds)/len_session: 如果测试集的最后一个分割不完整，则添加剩余的索引。
			test_inds_split.append(test_inds[-len_session:])
		
		test_id = list(range(len(test_inds_split)+len(train_inds_split)))[-len(test_inds_split):] 创建测试集的ID列表。

		train_idx[uid] = train_id[1:] 为当前用户设置训练集索引，跳过第一个ID。
		test_idx[uid] = test_id 为当前用户设置测试集索引
 
		for ind in train_id: 遍历训练集ID。

		#generate data for comparative methods such as deepmove
                这段代码是用于为比较方法（如DeepMove等）生成数据的Python脚本
			if ind == 0: 如果当前的索引 ind 为0，跳过当前循环迭代。
				continue

			data_train[uid][ind] = {} 为当前用户和当前索引创建一个空字典，用于存储训练数据。
			history_ind =[] 初始化一个空列表，用于存储用户的历史会话索引。
			for i in range(ind): 遍历从0到当前索引 ind 的所有索引。
				history_ind.extend(train_inds_split[i]) 将第i个会话的索引添加到历史会话索引列表中。
			whole_ind = [] 初始化一个空列表，用于存储整个会话的索引。
			whole_ind.extend(history_ind) 将历史会话索引添加到整个会话索引列表中。
			whole_ind.extend(train_inds_split[ind]) 将当前会话的索引添加到整个会话索引列表中。

			whole_data = group.loc[whole_ind] 根据整个会话的索引，从原始数据中选取相应的数据。
			loc = whole_data['venueid'].values[:-1] 提取整个会话中的位置ID，并去掉最后一个位置ID。
			tim = whole_data['hour'].values[:-1]  提取整个会话中的小时信息，并去掉最后一个小时
			target = group.loc[train_inds_split[ind][1:]]['venueid'].values 提取当前会话中下一个位置ID作为目标位置

			#loc = group_i['venueid'].values[:-1] 
			#tim = get_day(group_i['time'].values)[1][:-1]
			#target = group_i['venueid'].values[-10:]

			data_train[uid][ind]['loc'] = torch.LongTensor(loc).unsqueeze(1) 将位置ID转换为张量，并增加一个维度，存储在训练数据中。
			data_train[uid][ind]['tim'] = torch.LongTensor(tim).unsqueeze(1) 将小时信息转换为张量，并增加一个维度，存储在训练数据中
			data_train[uid][ind]['target'] = torch.LongTensor(target) 将目标位置ID转换为张量，存储在训练数据中。

			user_lastid[uid].append(loc[-1]) 将当前会话的最后一个位置ID添加到用户最后访问的位置ID列表中。

			group_i = group.loc[train_inds_split[ind]] 根据当前会话的索引，从原始数据中选取当前会话的数据。
		
			#generate data for SHAN 这段代码是用于为SHAN方法和开发者自己的方法生成数据的Python脚本
			current_loc = group_i['venueid'].values 提取当前会话的位置ID。
			data_train[uid][ind]['current_loc'] = torch.LongTensor(current_loc).unsqueeze(1) 将当前会话的位置ID转换为张量，并增加一个维度，存储在训练数据中。
			#group_i = whole_data 
		#generate data for my methods. X,Y,time,userid 
			
			current_cat = group_i['catid'].values 提取当前会话的类别ID。
			train_x.append(current_loc[:-1]) 将当前会话的位置ID（除了最后一个）添加到训练数据的特征列表中。
			train_x_cat.append(current_cat[:-1]) 将当前会话的类别ID（除了最后一个）添加到训练数据的特征列表中
			train_y.append(current_loc[1:]) 将当前会话的位置ID（除了第一个）添加到训练数据的目标列表中。
			#train_hour.append(group_i['hour_48'].values[:-1]) 
			train_hour.append(group_i['hour'].values[:-1]) 提取当前会话的小时信息，并添加到训练数据的特征列表中。
			train_userid.append(uid) 将用户ID添加到训练数据的用户ID列表中。
			
			#train_hour_pre.append(group_i['hour'].values[-1]) 
			#train_week_pre.append(group_i['week'].values[-1])

			train_hour_pre.append(group_i['hour'].values[1:]) 提取当前会话的小时信息（除了第一个），并添加到训练数据的小时预测特征列表中。
			train_week_pre.append(group_i['week'].values[1:]) 提取当前会话的周信息（除了第一个），并添加到训练数据的周预测特征列表中。

			train_indexs.append(group_i.index.values) 将当前会话的索引添加到训练数据的索引列表中。

		for ind in test_id: 遍历测试数据的索引。

			data_test[uid][ind] = {} 为当前用户和当前索引创建一个空字典，用于存储测试数据。
			history_ind =[] 初始化一个空列表，用于存储用户的历史会话索引
			for i in range(len(train_inds_split)): 遍历训练数据的会话索引。
				history_ind.extend(train_inds_split[i]) 将每个训练会话的索引添加到历史会话索引列表中。
			whole_ind = [] 初始化一个空列表，用于存储整个会话的索引。
			whole_ind.extend(history_ind) 将历史会话索引添加到整个会话索引列表中。
			whole_ind.extend(test_inds_split[ind-len(train_inds_split)]) 将当前测试会话的索引添加到整个会话索引列表中。

			whole_data = group.loc[whole_ind] 根据整个会话的索引，从原始数据中选取相应的数据。

			loc = whole_data['venueid'].values[:-1] 提取整个会话中的位置ID，并去掉最后一个位置ID。
			tim = whole_data['hour'].values[:-1] 提取整个会话中的小时信息，并去掉最后一个小时。
			target = group.loc[test_inds_split[ind-len(train_inds_split)][1:]]['venueid'].values 提取当前测试会话中下一个位置ID作为目标位置。

			#loc = group_i['venueid'].values[:-1] 
			#tim = get_day(group_i['time'].values)[1][:-1]
			#target = group_i['venueid'].values[-10:]

			data_test[uid][ind]['loc'] = torch.LongTensor(loc).unsqueeze(1) 将位置ID转换为张量，并增加一个维度，存储在测试数据中。
			data_test[uid][ind]['tim'] = torch.LongTensor(tim).unsqueeze(1) 将小时信息转换为张量，并增加一个维度，存储在测试数据中。
			data_test[uid][ind]['target'] = torch.LongTensor(target) 将目标位置ID转换为张量，存储在测试数据中。

			user_lastid[uid].append(loc[-1]) 将当前会话的最后一个位置ID添加到用户最后访问的位置ID列表中。

			#group_i = whole_data

			group_i = group.loc[test_inds_split[ind-len(train_inds_split)]] 根据当前测试会话的索引，从原始数据中选取当前测试会话的数据。

			current_loc = group_i['venueid'].values 提取当前测试会话的位置ID。
			data_test[uid][ind]['current_loc'] = torch.LongTensor(current_loc).unsqueeze(1) 将当前测试会话的位置ID转换为张量，并增加一个维度，存储在测试数据中。
			current_cat = group_i['catid'].values 提取当前测试会话的类别ID。
			test_x_cat.append(current_cat[:-1]) 将当前测试会话的类别ID（除了最后一个）添加到测试数据的特征列表中
			test_x.append(current_loc[:-1]) 将当前测试会话的位置ID（除了最后一个）添加到测试数据的特征列表中。
			test_y.append(current_loc[1:]) 将当前测试会话的位置ID（除了第一个）添加到测试数据的目标列表中。
			#test_hour.append(group_i['hour_48'].values[:-1]) 
			test_hour.append(group_i['hour'].values[:-1]) 提取当前测试会话的小时信息，并添加到测试数据的特征列表中。
			test_userid.append(uid) 将用户ID添加到测试数据的用户ID列表中。

			#test_hour_pre.append(group_i['hour'].values[-1]) 
			#test_week_pre.append(group_i['week'].values[-1])

			test_hour_pre.append(group_i['hour'].values[1:]) 提取当前测试会话的小时信息（除了第一个），并添加到测试数据的小时预测特征列表中。
			test_week_pre.append(group_i['week'].values[1:]) 提取当前测试会话的周信息（除了第一个），并添加到测试数据的周预测特征列表中。
 
			test_indexs.append(group_i.index.values) 将当前测试会话的索引添加到测试数据的索引列表中。


	with open('data_train.pk','wb') as f: 用 pickle 库将 data_train 字典保存到文件 data_train.pk 中。
		pickle.dump(data_train,f)

	with open('data_test.pk','wb') as f:
		pickle.dump(data_test,f)

	with open('train_idx.pk','wb') as f:
		pickle.dump(train_idx,f)

	with open('test_idx.pk','wb') as f:
		pickle.dump(test_idx,f)

	print('user_num: ',len(data_train.keys())) 打印训练数据中用户的数量。
	#minMax = MinMaxScaler()注释掉的代码是用于对长期特征进行最小最大归一化的，但这部分代码被注释掉了，所以不会执行
	#long_term_feature = minMax.fit_transform(np.array(long_term_feature))

	with open('long_term.pk','wb') as f: 将 long_term 字典保存到文件 long_term.pk 中。
		pickle.dump(long_term,f)

	#with open('long_term_feature.pk','wb') as f: 注释掉的代码是用于保存长期特征的，但这部分代码被注释掉了，所以不会执行。
	#	pickle.dump(long_term_feature,f)
        定义一个 dataloader 函数，该函数接受多个参数并创建一个 TensorDataset，然后使用 DataLoader 来批量加载数据，是否打乱数据由 shuffle 参数控制。
 	def dataloader(X,X_cat,Y,hour,userid,hour_pre,week_pre):
		
		torch_dataset = Data.TensorDataset(torch.LongTensor(X),torch.LongTensor(X_cat),torch.LongTensor(Y),torch.LongTensor(hour),torch.LongTensor(userid),torch.LongTensor(hour_pre),torch.LongTensor(week_pre))
		loader = Data.DataLoader(
			dataset = torch_dataset,  
			batch_size = batch_size,  
			shuffle = True,
			num_workers = 0,
		)
		return loader
         使用 dataloader 函数创建训练数据加载器 loader_train。
	loader_train = dataloader(train_x,train_x_cat,train_y,train_hour,train_userid,train_hour_pre,train_week_pre)
        使用 dataloader 函数创建测试数据加载器 loader_test。
	loader_test = dataloader(test_x,test_x_cat,test_y,test_hour,test_userid,test_hour_pre,test_week_pre)
        创建一个 pre_data 字典，包含数据的大小信息、训练数据加载器和测试数据加载器。
	pre_data = {}
	pre_data['size'] = [vocab_size_poi,vocab_size_cat,vocab_size_user,len(train_x),len(test_x)]
	pre_data['loader_train'] = loader_train
	pre_data['loader_test'] = loader_test
        将 pre_data 字典保存到文件 pre_data.txt 中。
	with open('pre_data.txt','wb') as f:
		pickle.dump(pre_data,f)
	return pre_data
