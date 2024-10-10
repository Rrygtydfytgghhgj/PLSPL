
import torch.nn as nn
import torch
from torch.autograd import Variable  
from torch.nn.parameter import Parameter
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pandas as pd
import numpy as np
定义了一个名为long_short的类，它继承自PyTorch的nn.Module，这是一个构建神经网络模型的基类。
class long_short(nn.Module):
	这是long_short类的构造函数，它初始化模型的各个参数和层。参数包括用户、地点兴趣点（POI）、类别、小时和周的嵌入向量大小，隐藏层大小，LSTM层数，各类别的词汇表大小，长期记忆参数，以及类别候选列表。
	def __init__(self,embed_size_user, embed_size_poi, embed_size_cat,embed_size_hour, embed_size_week,
		hidden_size, num_layers, vocab_poi, vocab_cat,vocab_user,vocab_hour,long_term,cat_candi):    

		super(long_short, self).__init__() 调用基类nn.Module的构造函数。
		
		self.embed_user = nn.Embedding(vocab_user, embed_size_user) 初始化一个嵌入层，用于将用户ID映射到嵌入向量
		self.embed_poi = nn.Embedding(vocab_poi, embed_size_poi) 初始化一个嵌入层，用于将地点兴趣点ID映射到嵌入向量。
		self.embed_cat = nn.Embedding(vocab_cat, embed_size_cat) 初始化一个嵌入层，用于将类别ID映射到嵌入向量
		self.embed_hour = nn.Embedding(vocab_hour, embed_size_hour) 初始化一个嵌入层，用于将小时ID映射到嵌入向量
		self.embed_week = nn.Embedding(7, embed_size_week) 初始化一个嵌入层，用于将一周中的天（0到6）映射到嵌入向量。
		
		self.embed_total_size = embed_size_poi + embed_size_cat + embed_size_hour + embed_size_week 计算总的嵌入向量大小，这是POI、类别、小时和周嵌入向量大小的总和。
		self.vocab_poi = vocab_poi 将构造函数参数 vocab_poi 的值赋给类的属性 self.vocab_poi。vocab_poi 通常表示地点（POI）的词汇表大小，即不同地点的数量。
		self.vocab_hour = vocab_hour 将构造函数参数 vocab_hour 的值赋给类的属性 self.vocab_hour。vocab_hour 通常表示小时的词汇表大小，即不同小时的数量。
		self.vocab_week = 7 将整数 7 赋给类的属性 self.vocab_week。这里因为一周有7天，所以直接将 7 作为星期的词汇表大小。
		self.long_term = long_term  将构造函数参数 long_term 的值赋给类的属性 self.long_term。long_term 参数可能用于表示模型中的长期记忆或长期依赖关系的处理方式。


		self.weight_poi = Parameter(torch.ones(embed_size_poi,embed_size_user))  定义了一个参数，用于加权用户和地点嵌入向量的点积
		self.weight_cat = Parameter(torch.ones(embed_size_cat,embed_size_user)) 定义了一个参数，用于加权用户和类别嵌入向量的点积
		self.weight_time = Parameter(torch.ones(embed_size_hour + embed_size_week,embed_size_user)) 定义了一个参数，用于加权用户和时间信息（小时和星期）嵌入向量的点积。
		self.bias = Parameter(torch.ones(embed_size_user)) 定义了一个偏置项，用于最后的输出计算。
		
		self.activate_func = nn.ReLU() 定义了激活函数，这里使用的是ReLU。

		self.num_layers = num_layers 保存了LSTM层的数量和隐藏层的大小。
		self.hidden_size = hidden_size
		
		self.out_w_long = Parameter(torch.Tensor([0.5]).repeat(vocab_user)) 定义了一个参数，用于加权长期记忆的输出。
		self.out_w_poi = Parameter(torch.Tensor([0.25]).repeat(vocab_user)) 定义了一个参数，用于加权地点嵌入的输出。
		self.out_w_cat = Parameter(torch.Tensor([0.25]).repeat(vocab_user)) 定义了一个参数，用于加权类别嵌入的输出。

		
		#self.w1 = Parameter(torch.Tensor([0.5])) 
		#self.w2 = Parameter(torch.Tensor([0.5]))
		#self.w3 = Parameter(torch.Tensor([0.5]))

		self.weight_hidden_poi = Parameter(torch.ones(self.hidden_size,1))  这行代码创建了一个形状为 (self.hidden_size, 1) 的参数矩阵 weight_hidden_poi。这个矩阵被初始化为全1的张量，然后通过 Parameter 封装，使其成为一个可学习的参数。这个参数可能用于加权LSTM层输出的地点（POI）相关信息，用于后续的计算或预测。
		self.weight_hidden_cat = Parameter(torch.ones(self.hidden_size,1)) 创建了一个形状为 (self.hidden_size, 1) 的参数矩阵 weight_hidden_cat。这个矩阵也被初始化为全1的张量，并通过 Parameter 封装，使其成为一个可学习的参数。这个参数可能用于加权LSTM层输出的类别（Category）相关信息，用于后续的计算或预测。

		self.vocab_poi = vocab_poi  将构造函数参数 vocab_poi 的值赋给类的属性 self.vocab_poi。这表示地点（POI）的数量。
		self.vocab_cat = vocab_cat  将构造函数参数 vocab_cat 的值赋给类的属性 self.vocab_cat。这表示类别（Category）的数量。
		size = embed_size_poi + embed_size_user + embed_size_hour 计算LSTM层输入的大小，它是地点嵌入大小、用户嵌入大小和小时嵌入大小的总和。
		size2 = embed_size_cat + embed_size_user + embed_size_hour  计算另一个LSTM层输入的大小，它是类别嵌入大小、用户嵌入大小和小时嵌入大小的总和。

		self.lstm_poi = nn.LSTM(size, hidden_size, num_layers, dropout = 0.5, batch_first = True) 初始化一个LSTM层，用于处理地点信息。输入大小是 size，隐藏层大小是 hidden_size，层数是 num_layers，dropout比率是0.5，且输入数据的批次维度在第一维（batch_first=True）
		self.lstm_cat = nn.LSTM(size2, hidden_size, num_layers, dropout = 0.5, batch_first = True) 初始化另一个LSTM层，用于处理类别信息。配置与地点的LSTM层相同。
		self.fc_poi = nn.Linear(hidden_size,self.vocab_poi) 初始化一个全连接层，将LSTM层的输出（地点信息）映射到地点的词汇表大小 vocab_poi。
		self.fc_cat = nn.Linear(hidden_size,self.vocab_poi) 初始化另一个全连接层，将LSTM层的输出（类别信息）映射到地点的词汇表大小 vocab_poi。这里可能是一个错误，通常我们期望它映射到类别的词汇表大小 vocab_cat。
		self.attn_linear = nn.Linear(self.hidden_size*2, self.vocab_poi) 初始化一个线性层，用于处理注意力机制的输出。输入大小是 hidden_size*2，输出大小是 vocab_poi。

		self.fc_longterm = nn.Linear(self.embed_total_size, self.vocab_poi) 初始化一个全连接层，用于处理长期记忆信息。输入大小是 embed_total_size，输出大小是 vocab_poi。

	def forward(self, inputs_poi, inputs_cat, inputs_user,inputs_time,hour_pre,week_pre,poi_candi,cat_candi):#,inputs_features):定义了模型的前向传播函数，它接收多个输入参数。inputs_poi：地点（POI）的输入数据。inputs_cat：类别（Category）的输入数据。inputs_user：用户的输入数据。inputs_time：时间的输入数据。hour_pre 和 week_pre：可能是预先处理好的小时和星期的嵌入向量。poi_candi 和 cat_candi：地点和类别的候选集合。


		out_poi = self.get_output(inputs_poi,inputs_user,inputs_time,self.embed_poi,self.embed_user,self.embed_hour,self.lstm_poi,self.fc_poi) 调用 get_output 方法来处理地点（POI）相关的输入数据。这个方法可能包括嵌入查找、LSTM处理和全连接层处理的步骤，最终返回地点的输出。
		out_cat = self.get_output(inputs_cat,inputs_user,inputs_time,self.embed_cat,self.embed_user,self.embed_hour,self.lstm_cat,self.fc_cat) 类似地，调用 get_output 方法来处理类别（Category）相关的输入数据，并返回类别的输出。


#########################################################################################

		# long-term preference 
		
		u_long = self.get_u_long(inputs_user) 调用 get_u_long 方法来获取用户的长期偏好。

		# with fc_layer
		out_long = self.fc_longterm(u_long).unsqueeze(1).repeat(1,out_poi.size(1),1) 将用户的长期偏好通过一个全连接层 fc_longterm 处理，然后使用 unsqueeze 方法在第二个维度（索引为1）增加一个维度，使其形状变为 [batch_size, 1, vocab_poi]。接着使用 repeat 方法将这个向量在第二个维度（索引为1）上重复 out_poi.size(1) 次，以匹配 out_poi 的序列长度。这样，每个时间步的长期偏好都被复制了相同的次数，以便可以在后续的计算中与每个时间步的短期偏好相结合。

#########################################################################################
	#output 

	# weighted sum directly  
	这段代码是模型前向传播过程中的一部分，它展示了如何将用户的长期偏好、地点（POI）偏好和类别（Category）偏好结合起来，以生成最终的输出
		weight_poi = self.out_w_poi[inputs_user]  通过用户的输入 inputs_user 索引到模型的一个参数 self.out_w_poi，获取每个用户的地点偏好权重
		weight_cat = self.out_w_cat[inputs_user]  似地，这行代码获取每个用户的类别偏好权重，表示模型预测用户对类别的兴趣程度。
		#weight_long = self.out_w_long[inputs_user] #32  
		weight_long = 1-weight_poi-weight_cat weight_long = 1 - weight_poi - weight_cat：这行代码计算长期偏好的权重。这里假设长期偏好、地点偏好和类别偏好的权重之和为1，即 weight_long 是剩余的权重部分，表示长期偏好对用户兴趣的贡献。

	
		out_poi = torch.mul(out_poi, weight_poi.unsqueeze(1).repeat(1,19).unsqueeze(2)) 这行代码将 out_poi（地点的预测输出）与地点偏好权重 weight_poi 相乘。weight_poi 通过 unsqueeze 和 repeat 操作调整形状，以匹配 out_poi 的维度，确保可以进行逐元素乘法。
		out_cat = torch.mul(out_cat, weight_cat.unsqueeze(1).repeat(1,19).unsqueeze(2)) 类似地，这行代码将 out_cat（类别的预测输出）与类别偏好权重 weight_cat 相乘。
		out_long = torch.mul(out_long, weight_long.unsqueeze(1).repeat(1,19).unsqueeze(2)) 这行代码将 out_long（长期的预测输出）与长期偏好权重 weight_long 相乘。
		
		out = out_poi + out_cat + out_long 这行代码将加权后的地点偏好、类别偏好和长期偏好相加，得到最终的输出 out。这个输出综合了用户的长期和短期偏好，可以用于生成更准确的推荐。

		return out 这行代码返回最终的输出 out，它将被用于后续的损失计算或作为推荐系统的输出。

	def get_u_long(self,inputs_user): 定义了一个名为 get_u_long 的方法，它用于计算用户的长期偏好向量
		# get the hidden vector of users' long-term preference 

		u_long = {} 初始化一个空字典 u_long 来存储每个用户的长期偏好向量。
		for user in inputs_user: 遍历输入的用户ID列表。

			user_index = user.tolist() 将用户ID转换为列表形式（通常是一个整数）。
			if user_index not in u_long.keys(): 如果该用户尚未被处理，则进行以下操作。
                                     提取长期数据
				poi = self.long_term[user_index]['loc'] 从长期记忆数据中提取该用户的位置（POI）历史。
				hour = self.long_term[user_index]['hour'] 提取用户活跃的小时历史
				week = self.long_term[user_index]['week'] 提取用户活跃的星期历史
				cat = self.long_term[user_index]['category'] 提取用户感兴趣的类别历史。
                                    嵌入层：
				seq_poi = self.embed_poi(poi) 将POI历史转换为嵌入向量。
				seq_cat = self.embed_cat(cat) 将类别历史转换为嵌入向量。
				seq_user = self.embed_user(user) 将用户ID转换为嵌入向量。
				seq_hour = self.embed_hour(hour) 将小时历史转换为嵌入向量。
				seq_week = self.embed_week(week) 将星期历史转换为嵌入向量
				seq_time = torch.cat((seq_hour, seq_week),1) 将小时和星期的嵌入向量拼接在一起
                                加权求和：
				poi_mm = torch.mm(seq_poi, self.weight_poi)计算POI嵌入向量的加权和
				cat_mm = torch.mm(seq_cat, self.weight_cat) 计算类别嵌入向量的加权和。
				time_mm = torch.mm(seq_time, self.weight_time) 计算时间嵌入向量的加权和。
                                 激活函数：
				hidden_vec =  poi_mm.add_(cat_mm).add_(time_mm).add_(self.bias)将加权和的结果与偏置相加。
				hidden_vec = self.activate_func(hidden_vec)# 876*50 应用激活函数（如ReLU）。
				alpha = F.softmax(torch.mm(hidden_vec, seq_user.unsqueeze(1)),0) #876*1  计算注意力权重，表示每个历史记录对用户长期偏好的贡献。

				poi_concat = torch.cat( (seq_poi,seq_cat, seq_hour, seq_week), 1) #876*427 将所有嵌入向量拼接在一起。

				u_long[user_index] = torch.sum( torch.mul(poi_concat, alpha),0 ) 计算加权的长期偏好向量。

		构建长期偏好向量张量：
		u_long_all = torch.zeros(len(inputs_user),self.embed_total_size).cuda() 初始化一个张量来存储所有用户的长期偏好向量。
		#64*427
		for i in range(len(inputs_user)): 遍历所有用户。
			u_long_all[i,:] = u_long[inputs_user.tolist()[i]] 将计算得到的长期偏好向量存储到张量中。
		
		return u_long_all 返回包含所有用户长期偏好向量的张量。

             定义了一个名为 get_output 的方法，它用于处理输入序列，并通过嵌入层、LSTM层和全连接层来生成输出
             inputs：要处理的输入序列（例如，地点ID序列），inputs_user：用户的输入数据（例如，用户ID），inputs_time：时间的输入数据（例如，小时和星期）。embed_layer：用于将输入序列转换为嵌入向量的嵌入层。embed_user：用于将用户ID转换为嵌入向量的嵌入层。embed_time：用于将时间信息转换为嵌入向量的嵌入层。lstm_layer：用于处理序列数据的LSTM层。fc_layer：用于将LSTM层的输出转换为最终预测的全连接层。
	def get_output(self, inputs,inputs_user,inputs_time,embed_layer,embed_user,embed_time,lstm_layer,fc_layer): 

			# embed your sequences
		seq_tensor = embed_layer(inputs) 将输入序列 inputs 通过嵌入层 embed_layer 转换为嵌入向量。
		seq_user = embed_user(inputs_user).unsqueeze(1).repeat(1,seq_tensor.size(1),1) 将用户ID inputs_user 通过嵌入层 embed_user 转换为嵌入向量，并调整形状以匹配序列长度
		seq_time = embed_time(inputs_time)将时间信息 inputs_time 通过嵌入层 embed_time 转换为嵌入向量。
			# embed your sequences
		input_tensor = torch.cat((seq_tensor,seq_user,seq_time),2)将序列嵌入向量、用户嵌入向量和时间嵌入向量在特征维度上拼接在一起，形成输入张量。
			# pack them up nicely
		output, _ = lstm_layer(input_tensor)将输入张量通过LSTM层 lstm_layer 处理，得到输出序列。这里使用 torch.cat 拼接的张量作为LSTM层的输入，LSTM层的输出包括最后一个时间步的隐藏状态和最后一个时间步的输出。
		out = fc_layer(output) # the last outputs 将LSTM层的输出通过全连接层 fc_layer 处理，得到最终的预测输出。这里通常使用LSTM层的最后一个时间步的输出作为全连接层的输入
		return out 返回最终的预测输出 out


