import random
from model import news
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 交叉验证
cross_validation_list= []
lambda_list = [0.0001,0.001,0.01,0.05,0.125,0.25,0.5,0.75,1,2] #构造超参数搜索空间
for hyper_param in lambda_list:
    temp_acc = 0
    temp_acc_list = []
    for fold in range(0,5):
        rand_state = random.randrange(0,100) #设置随机的随机状态
        x_train, x_test, y_train, y_test = train_test_split(news.data,news.target,test_size=0.2,random_state=rand_state)
        vec = CountVectorizer()
        x_train = vec.fit_transform(x_train)
        x_test = vec.transform(x_test)
        # 初始化朴素贝叶斯模型
        multinomial_naive_bayes = MultinomialNB(alpha=hyper_param)
        # 训练集合上进行训练， 估计参数
        multinomial_naive_bayes.fit(x_train, y_train)
        # 对测试集合进行预测 保存预测结果
        y_predict = multinomial_naive_bayes.predict(x_test)
        fold_accuracy = multinomial_naive_bayes.score(x_test, y_test)
        print("validation accuracy:", fold_accuracy ,"fold",fold+1,"lambda=",hyper_param)
        temp_acc+=fold_accuracy
        temp_acc_list.append(fold_accuracy)
    print("lambda:",hyper_param,"accuracy:",temp_acc/5)
    cross_validation_list.append(temp_acc_list)
