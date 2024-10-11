# long-term using attention

import pandas as pd
import time
import datetime
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.utils.data as Data
from torch.backends import cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
import sys
import codecs

import preprocess_longshort as preprocess
import model_longshort as model

SEED = 0  设置随机种子为0，以确保实验的可重复性。
torch.manual_seed(SEED) 为PyTorch的随机数生成器设置种子。
torch.cuda.manual_seed(SEED) 为PyTorch的CUDA（GPU加速）随机数生成器设置种子。


run_name = "user-aware-0.001-NYC" 定义运行名称，用于保存日志和模型。

log = open("log_" + run_name + ".txt", "w") 打开一个日志文件，用于记录程序的输出。
sys.stdout = log 将标准输出重定向到上面打开的日志文件。
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 这行代码被注释掉了，如果取消注释，它会指定程序只使用编号为0的GPU。

# Training Parameters
batch_size = 32 设置批量大小为32。
hidden_size = 128 设置隐藏层的大小为128。
num_layers = 1 设置网络层数为1。
num_epochs = 25 设置训练的轮数为25。
lr = 0.001 设置学习率为0.001。

vocab_hour = 24 设置小时的词汇表大小为24（一天的小时数）。
vocab_week = 7  设置一周的词汇表大小为7（一周的天数）。

embed_poi = 300 设置地点嵌入的维度为300。
embed_cat = 100 设置类别嵌入的维度为100。
embed_user = 50 设置用户嵌入的维度为50。
embed_hour = 20 设置小时嵌入的维度为20。
embed_week = 7 设置周嵌入的维度为7。
打印出嵌入维度和学习率等参数。
print("emb_poi :", embed_poi) 
print("emb_user :", embed_user)
print("hidden_size :", hidden_size)
print("lr :", lr)

data = pd.DataFrame(pd.read_table("../input/dataset_TSMC2014_NYC.txt", header=None, encoding="latin-1")) 读取数据集并创建一个DataFrame。
data.columns = ["userid", "venueid", "catid", "catname", "latitute", "longitude", "timezone", "time"] 为DataFrame设置列名。

print("start preprocess") 打印开始预处理数据的信息。
#pre_data = preprocess.sliding_varlen(data, batch_size) 这行代码被注释掉了，如果取消注释，它会调用一个函数来预处理数据。
print("pre done") 打印预处理完成的信息。

使用pickle库加载预处理后的数据、长期特征和类别候选项。
with open("pre_data.txt", "rb") as f:
    pre_data = pickle.load(f)

with open("long_term.pk", "rb") as f:
    long_term = pickle.load(f)

with open("cat_candidate.pk", "rb") as f:
    cat_candi = pickle.load(f)

# with open('long_term_feature.pk','rb') as f:
# 	long_term_feature = pickle.load(f)
long_term_feature = [0] 定义一个包含0的列表作为长期特征的占位符。

cat_candi = torch.cat((torch.Tensor([0]), cat_candi)) 将0添加到类别候选项的Tensor前面。
cat_candi = cat_candi.long() 将类别候选项转换为长整型。

[vocab_poi, vocab_cat, vocab_user, len_train, len_test] = pre_data["size"] 

loader_train = pre_data["loader_train"] 获取训练和测试数据的加载器。
loader_test = pre_data["loader_test"] 

print("train set size: ", len_train) 打印训练集和测试集的大小，以及词汇表的大小。
print("test set size: ", len_test)
print("vocab_poi: ", vocab_poi)
print("vocab_cat: ", vocab_cat)

print("Train the Model...")


Model = model.long_short( 这行代码实例化了一个名为 long_short 的模型类，这个模型类可能是一个结合了长期和短期特征的模型
    embed_user, 用户嵌入的维度。
    embed_poi, 兴趣点（POI）嵌入的维度。
    embed_cat,  类别嵌入的维度。
    embed_hour, 小时嵌入的维度。
    embed_week, 周嵌入的维度。
    hidden_size, 隐藏层的大小。
    num_layers, 网络层数。
    vocab_poi + 1, POI词汇表的大小加1（通常用于表示未知词）。
    vocab_cat + 1, 类别词汇表的大小加1。
    vocab_user + 1, 用户词汇表的大小加1
    vocab_hour, 小时词汇表的大小。
    long_term, 长期特征。
    cat_candi, 类别候选项。
    # len(long_term_feature[0]), 这行代码被注释掉了，如果取消注释，它会传入长期特征的长度。
)



userid_cursor = False 定义了一个布尔变量 userid_cursor 并将其设置为 False。这个变量可能用于控制数据加载或结果输出的逻辑，但在这里它被禁用。
results_cursor = False 定义了一个布尔变量 results_cursor 并将其设置为 False。这个变量可能用于控制数据加载或结果输出的逻辑，但在这里它被禁用。

Model = Model.cuda() 将模型移动到GPU上，以便利用CUDA加速训练过程。

loss_function = nn.CrossEntropyLoss() 定义了交叉熵损失函数，这是一种常用于分类任务的损失函数。
optimizer = optim.Adam(Model.parameters(), lr) 定义了Adam优化器，并将其应用于模型的参数，使用之前定义的学习率 lr。


def precision(indices, batch_y, k, count, delta_dist): 定义了一个计算精确度（precision）的函数。indices：预测结果的索引。batch_y：真实标签。k：要评估的前k个预测。count：总样本数。delta_dist：这个参数在函数中未使用。 函数内部，对于每个样本，检查真实标签是否在前k个预测中。如果是，则增加精确度计数器。最后，返回精确度计数器除以总样本数，得到平均精确度。
    precision = 0 初始化精确度计数器为0。
    for i in range(indices.size(0)): 遍历每个样本。
        sort = indices[i] 获取第i个样本的预测结果索引。
        if batch_y[i].long() in sort[:k]:检查真实标签是否在前k个预测中。
            precision += 1 如果是，则增加精确度计数器。
    return precision / count 返回平均精确度。


def MAP(indices, batch_y, k, count):  定义了一个计算平均精度均值（Mean Average Precision, MAP）的函数。indices：预测结果的索引。batch_y：真实标签。k：要评估的前k个预测。count：总样本数。
    sum_precs = 0  初始化精度均值计数器为0。
    for i in range(indices.size(0)): 遍历每个样本。
        sort = indices[i] 获取第i个样本的预测结果索引。
        ranked_list = sort[:k] 获取前k个预测。
        hists = 0 初始化历史计数器为0。
        for n in range(len(ranked_list)): 遍历前k个预测。
            if ranked_list[n].cpu().numpy() in batch_y[i].long().cpu().numpy(): 检查预测结果是否与真实标签匹配。
                hists += 1 如果是，则增加历史计数器。
                sum_precs += hists / (n + 1) 计算当前样本的精度均值，并累加到总精度均值计数器。
    return sum_precs / count 返回平均精度均值。


def recall(indices, batch_y, k, count, delta_dist): 定义了一个计算召回率（recall）的函数。indices：预测结果的索引。batch_y：真实标签。k：要评估的前k个预测。count：总样本数。delta_dist：这个参数在函数中未使用。
    recall_correct = 0  初始化召回率正确计数器为0。
    for i in range(indices.size(0)): 遍历每个样本。
        sort = indices[i] 获取第i个样本的预测结果索引。
        if batch_y[i].long() in sort[:k]: 检查真实标签是否在前k个预测中。
            recall_correct += 1 如果是，则增加召回率正确计数器。
    return recall_correct / count 返回平均召回率。


for epoch in range(num_epochs): 开始训练循环，num_epochs 是训练的轮数。
    Model = Model.train() 将模型设置为训练模式。
    total_loss = 0.0 初始化总损失为0。

    precision_1 = 0 初始化精确度和召回率的计数器，分别对应不同的k值（1, 5, 10, 20）。
    precision_5 = 0
    precision_10 = 0
    precision_20 = 0

    recall_5 = 0
    recall_10 = 0

    MAP_1 = 0 初始化MAP（平均精度均值）的计数器，分别对应不同的k值（1, 5, 10, 20）
    MAP_5 = 0
    MAP_10 = 0
    MAP_20 = 0

    userid_wrong_train = {} 初始化两个字典，可能用于记录在训练和测试中表现不佳的用户ID。
    userid_wrong_test = {}
    results_train = [] 初始化两个列表，用于存储训练和测试的结果。
    results_test = []

    for step, (batch_x, batch_x_cat, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_train): 遍历训练数据加载器 loader_train 中的批次。
        Model.zero_grad()  在反向传播前，将模型的梯度清零   
        users = batch_userid.cuda() 将用户ID数据移动到GPU上。
        hourids = Variable(hours.long()).cuda() 将小时ID数据转换为长整型并移动到GPU上。

        batch_x, batch_x_cat, batch_y, hour_pre, week_pre = (   将输入数据转换为变量并移动到GPU上。
            Variable(batch_x).cuda(),
            Variable(batch_x_cat).cuda(),
            Variable(batch_y).cuda(),
            Variable(hour_pre.long()).cuda(),
            Variable(week_pre.long()).cuda(),
        )

        poi_candidate = list(range(vocab_poi + 1)) 创建一个包含所有POI候选项的列表。
        poi_candi = Variable(torch.LongTensor(poi_candidate)).cuda() 将POI候选项转换为长整型张量并移动到GPU上。
        cat_candi = Variable(cat_candi).cuda() 将类别候选项转换为变量并移动到GPU上。
        outputs = Model(   将输入数据传递给模型，获取模型的输出。
            batch_x, batch_x_cat, users, hourids, hour_pre, week_pre, poi_candi, cat_candi
        )  

        loss = 0 初始化损失为
        for i in range(batch_x.size(0)):
            loss += loss_function(outputs[i, :, :], batch_y[i, :]).cuda()

        loss.backward()
        optimizer.step()

        total_loss += float(loss)

        outputs2 = outputs[:, -1, :]
        batch_y2 = batch_y[:, -1]

        out_p, indices = torch.sort(outputs2, dim=1, descending=True)
        count = float(len_train)
        delta_dist = 0
        precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
        precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
        precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
        precision_20 += precision(indices, batch_y2, 20, count, delta_dist)

        MAP_1 += MAP(indices, batch_y2, 1, count)
        MAP_5 += MAP(indices, batch_y2, 5, count)
        MAP_10 += MAP(indices, batch_y2, 10, count)
        MAP_20 += MAP(indices, batch_y2, 20, count)

    print(
        "train:",
        "epoch: [{}/{}]\t".format(epoch, num_epochs),
        "loss: {:.4f}\t".format(total_loss),
        "precision@1: {:.4f}\t".format(precision_1),
        "precision@5: {:.4f}\t".format(precision_5),
        "precision@10: {:.4f}\t".format(precision_10),
        "precision@20: {:.4f}\t".format(precision_20),
        "MAP@1: {:.4f}\t".format(MAP_1),
        "MAP@5: {:.4f}\t".format(MAP_5),
        "MAP@10: {:.4f}\t".format(MAP_10),
        "MAP@20: {:.4f}\t".format(MAP_20),
    )

    savedir = "checkpoint_file/checkpoint_" + run_name  
    if not os.path.exists(savedir):    
        os.makedirs(savedir)
    savename = savedir + "/checkpoint" + "_" + str(epoch) + ".tar"  

    torch.save({"epoch": epoch + 1, "state_dict": Model.state_dict(),}, savename) 
    
    if epoch % 1 == 0:

        Model = Model.eval()

        total_loss = 0.0

        precision_1 = 0
        precision_5 = 0
        precision_10 = 0
        precision_20 = 0

        MAP_1 = 0
        MAP_5 = 0
        MAP_10 = 0
        MAP_20 = 0

        for step, (batch_x, batch_x_cat, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_test):
            Model.zero_grad()
            hourids = hours.long()
            users = batch_userid

            batch_x, batch_x_cat, batch_y, hour_pre, week_pre = (
                Variable(batch_x).cuda(),
                Variable(batch_x_cat).cuda(),
                Variable(batch_y).cuda(),
                Variable(hour_pre.long()).cuda(),
                Variable(week_pre.long()).cuda(),
            )
            users = Variable(users).cuda()
            hourids = Variable(hourids).cuda()

            outputs = Model(
                batch_x, batch_x_cat, users, hourids, hour_pre, week_pre, poi_candi, cat_candi
            ) 
            loss = 0
            for i in range(batch_x.size(0)):
                loss += loss_function(outputs[i, :, :], batch_y[i, :])

            total_loss += float(loss)

            outputs2 = outputs[:, -1, :]
            batch_y2 = batch_y[:, -1]

            weights_output = outputs2.data

            outputs2 = weights_output  # +weights_classify# + weights_comatrix +weights_hour_prob
            out_p, indices = torch.sort(outputs2, dim=1, descending=True)

            count = float(len_test)

            precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
            precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
            precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
            precision_20 += precision(indices, batch_y2, 20, count, delta_dist)

            MAP_1 += MAP(indices, batch_y2, 1, count)
            MAP_5 += MAP(indices, batch_y2, 5, count)
            MAP_10 += MAP(indices, batch_y2, 10, count)
            MAP_20 += MAP(indices, batch_y2, 20, count)

        print(
            "val:",
            "loss: {:.4f}\t".format(total_loss),
            "precision@1: {:.4f}\t".format(precision_1),
            "precision@5: {:.4f}\t".format(precision_5),
            "precision@10: {:.4f}\t".format(precision_10),
            "precision@20: {:.4f}\t".format(precision_20),
            "MAP@1: {:.4f}\t".format(MAP_1),
            "MAP@5: {:.4f}\t".format(MAP_5),
            "MAP@10: {:.4f}\t".format(MAP_10),
            "MAP@20: {:.4f}\t".format(MAP_20),
        )


log.close()

