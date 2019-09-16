import json
import sys
import word2vec
import matplotlib.pyplot as plt
from pyhanlp import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import LSTM,GRU,ConvLSTM2D,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import regularizers
from keras.callbacks import EarlyStopping



def load_data():
	papers = []
	filename = sys.argv[1]
	file = open(filename,'r')
	for line in file.readlines():
		dic = json.loads(line)
		papers.append(dic)

	word_embedding = word2vec.load('word2vec_result.bin')#导入预训练好的word2vec embedding
	print('load data finished')
	return papers,word_embedding



def create_input(papers,word_embedding,max_length):

	#一行中有100个特征，即一个词用100维度向量表示
	#一共100行，即100个词语（时间步长）
	max_length
	word_matrix = []
	input = [] #训练集
	x_test = []#测试集
	exist=0
	for i in range(len(papers)):
		exist=0
		for item in papers[i]:
			if '认为' in item:#extract words from '认为'，本院认为中的信息重要
				exist = 1 #某条案件中是否存在本院认为，若不存在，则洗掉这条
				benyuanrenweis = HanLP.segment(papers[i][item])
				first_word=1
				keywords = HanLP.extractKeyword(papers[i][item],min(97,len(benyuanrenweis)))#本院认为中提取97个词
				#for keyword in keywords:
				#	print(keyword)
				for keyword in keywords:
					if keyword!='\n' and keyword!=' ':
						#构建每一个案件的二维输入矩阵
						if first_word == 1:
							word_matrix = word_embedding[keyword].reshape(1,-1)
							first_word=0
						else:
							word_matrix = np.row_stack((word_matrix,word_embedding[keyword].reshape(1,-1)))
		#extract word from 'docName'
		if exist==1:
			keywords = HanLP.extractKeyword(papers[i]['docName'],3) #docName中的信息也很重要，提取3个关键词
			for keyword in keywords:
				#print('docName:\n',keyword)
				if keyword!='\n' and keyword!=' ':
					word_matrix = np.row_stack((word_matrix,word_embedding[keyword].reshape(1,-1)))

			word_matrix = np.pad(word_matrix,((0,max_length-word_matrix.shape[0]),(0,0)),'mean')
			word_matrix = np.reshape(word_matrix,(1,100,max_length))
			#构建三维矩阵作为神经网络输入，2/3作为训练集，1/3作为测试集
			if i%3!=0:#训练集
				if i==1:#the first to put in input matrix
					input = word_matrix
				else:
					input = np.concatenate([input,word_matrix],0)
				print(input.shape)

			else :#测试集
				if i==0:
					x_test = word_matrix
				else:
					x_test = np.concatenate([x_test,word_matrix],0)

		if exist==0:
			#print(i)
			exist=1

	np.save("input_train100100_mean.npy",input)
	np.save("input_test100100_mean.npy",x_test)


def create_output(papers):
	# 给reason编号
	reasons = {}
	j=0
	flag=1
	reasonNum = [0 for i in range(180)]
	for i in range(len(papers)):
		if i==0:
			reasons[papers[i]['reason']]=j
			j=j+1
			#print(j,reasons)

		else:
			for k in range(i):
				if papers[i]['reason']==papers[k]['reason']:
					flag=0

			if flag==1:
				reasons[papers[i]['reason']]=j
				j=j+1
			flag=1
	#每个reason有多少个案例
	for i in range(len(papers)):
		for reason in reasons:
			if reason==papers[i]['reason']:
				reasonNum[reasons[reason]]+=1


	output_train = []
	output_test = []
	for i in range(len(papers)):
	#	if i!=527 and i!=800 and i!=881 and i!= 1011 and i!=1402 and i!=1410 and i!=1450 and i!=1457 and i!=1459 and i!=1473:#civil中这些案件没有“本院认为”或“一审法院认为”，剔除
	#	if i!=1638 and i!=2095 and i!=3262:#criminal中这些案件没有“本院认为”或“一审法院认为”，剔除
			if i%3!=0:
				output_train.append(reasons[papers[i]['reason']])
			else :
				output_test.append(reasons[papers[i]['reason']])
	output_train = np.array(output_train)
	output_train = np_utils.to_categorical(output_train,num_classes = len(reasons))
	output_test = np.array(output_test)
	output_test_c = np_utils.to_categorical(output_test,num_classes = len(reasons))
	#print (output)
	print ("ouput.shape: ",output_train.shape,output_test_c.shape)

	return output_train,output_test,output_test_c,reasons,reasonNum


def nerual_network(input_train,output_train,reasons):
	#neural network
	model = Sequential()
	model.add(GRU(384,input_shape=(100,100)))

	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,kernel_regularizer=regularizers.l2(0.01),activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=len(reasons),activation = 'softmax'))

	model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
	early_stopping = EarlyStopping(monitor='val_loss',patience=50,verbose=1,mode='auto')
	history = model.fit(input_train,output_train,validation_split=0.15,batch_size=500,epochs=500,callbacks=[early_stopping])

	model.save('model100100.h5')

	#可视化训练过程
	plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model acc')
	plt.ylabel('acc')
	plt.xlabel('epoch')
	plt.legend(['train','test'],loc='upper left')
	plt.savefig('./acc.png')

	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'],loc='upper right')
	plt.savefig('./loss.png')


def get_result(input_train,output_train,input_test,output_test,output_test_c,reasons,reasonNum):
	f=open('result.txt','a')
	model = load_model('model100100.h5')
	result = model.evaluate(input_train,output_train,batch_size=100)
	#print('\nTrain Acc:',result[1])
	f.write('\nTrain Acc:')
	f.write(str(result[1]))
	result = model.evaluate(input_test,output_test_c)
	#print('\nTest Acc:',result[1])
	f.write('\nTest Acc:')
	f.write(str(result[1]))

	print(model.summary())

	#print(np.argmax(model.predict(input_test),axis=1))
	predict = np.argmax(model.predict(input_test),axis=1)

	#打印判断错误的案件的真实案由和输出案由
	count=0
	for j in range(predict.shape[0]):
		if predict[j]!=output_test[j]:
			count=count+1
			#print(j)
			for reason in reasons:
				if reasons[reason]==output_test[j]:
					#print('correct:',reasons[reason],reason)
					correct_reason = reason
				if reasons[reason]==predict[j]:
					#print('predict:',reasons[reason],reason)
					predict_reason = reason
			f.write('\n')
			f.write('correct reason: ')
			f.write(correct_reason)
			f.write(' predict reason: ')
			f.write(predict_reason)
	print('total error:',count,count/predict.shape[0])


	num_of_this_reason_in_test=[0 for i in range(len(reasons))]
	num_of_this_reason_error_in_test=[0 for i in range(len(reasons))]

	#对于每种案由的具体分析
	#以下格式： 案由 数据集中该案由案件数目 测试集中该案由案件数目 测试集中该案由预测错误数目 测试率（测试数目/数据集中总数目） 错误率
	f.write('\n')
	for reason in reasons:
		f.write(reason)
		#f.write('\tnum of cases: ')
		f.write('\t')
		f.write(str(reasonNum[reasons[reason]]))
		for i in range(predict.shape[0]):
			if reasons[reason]==output_test[i]:
				num_of_this_reason_in_test[reasons[reason]]+=1
				if output_test[i]!=predict[i]:
					num_of_this_reason_error_in_test[reasons[reason]]+=1
		#f.write(' total number in test: ')
		f.write('\t')
		f.write(str(num_of_this_reason_in_test[reasons[reason]]))
		#f.write(' error number in test: ')
		f.write('\t')
		f.write(str(num_of_this_reason_error_in_test[reasons[reason]]))
		test_rate=num_of_this_reason_in_test[reasons[reason]]/reasonNum[reasons[reason]]
		#f.write(' test rate: ')
		f.write('\t')
		f.write(str(test_rate))
		if num_of_this_reason_in_test[reasons[reason]]!=0:
			#f.write(' error rate: ')
			f.write('\t')
			f.write(str(num_of_this_reason_error_in_test[reasons[reason]]/num_of_this_reason_in_test[reasons[reason]]))
		f.write('\n')
	f.close()

def split_word():
    #分词用于训练embedding
    f = open('split_words.txt','a')
    for i in range(len(papers)):
	    for word in papers[i]:
		    if word!='id' and word!='caseNo' and word!='judgementDateStart':
			    splits = HanLP.segment(papers[i][word])
			    for split in splits:
				    f.write(split.word)
				    f.write('\t')
			    f.write('\n')


def main():

	max_length = 100
	papers, word_embedding = load_data()
	create_input(papers,word_embedding, max_length)
	input_train = np.load("input_train100100_mean.npy")
	input_test = np.load("input_test100100_mean.npy")
	#print (input_train,input_test)
	output_train, output_test, output_test_c, reasons, reasonNum = create_output(papers)
	nerual_network(input_train,output_train,reasons)
	get_result(input_train,output_train,input_test,output_test,output_test_c,reasons,reasonNum)
	print('success!!')

if __name__=='__main__':
	main()


# 案由个数
# criminal 137
# civil 180
# admin 36
'''
id
docName		*
court
caseNo
caseType
instrumentType
reason
procedureId
referenceType
judgementDateStart
当事人
审理经过
公诉机关称
本院查明
本院认为	*
裁判结果
审判人员
'''
