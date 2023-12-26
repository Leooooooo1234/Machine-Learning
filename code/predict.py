from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from model import news

# 预测
x_train, x_test, y_train, y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=3)
#3 贝叶斯分类器对新闻进行预测
# 进行文本转化为特征
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
# 初始化朴素贝叶斯模型
mnb = MultinomialNB()
# 训练集合上进行训练， 估计参数
mnb.fit(x_train, y_train)
# 对测试集合进行预测 保存预测结果
y_predict = mnb.predict(x_test)
#模型评估
print("Accuracy:", mnb.score(x_test, y_test))
print("Indicators：\n",classification_report(y_test, y_predict, target_names=news.target_names))

