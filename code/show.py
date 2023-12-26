import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cross_validation import cross_validation_list
from cross_validation import lambda_list

# 结果可视化
cv_mean=[] # 交叉验证得到的均值
for lst in cross_validation_list:
    cv_mean.append(np.mean(lst))

df = pd.DataFrame()
df['Lambda'] = lambda_list
df['Classification Accuracy'] =  cv_mean
plt.figure(figsize=(10, 4))
plt.ylim((0.5,1))
my_y_ticks = np.arange(0.5,1,0.05)
plt.yticks(my_y_ticks)
p1=sns.barplot( data=df, x='Lambda', y='Classification Accuracy',palette="Blues_d")
plt.show()
