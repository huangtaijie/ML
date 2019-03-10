# 线性回归

## 问题描述



# 一辆二手车的价格取决于品牌和型号、车型年份、行驶里程、车况等诸多因素，而不管该车是从经销商处还是从私人卖家处购买。为了调查汽车里程数与销售价格之间的关系，我们收集了10辆本田雅阁2000款私家车的行驶里程与销售价格的数据：

| 行驶里程（1000英里） | 价格   （1000美元） |
| -------------------- | ------------------- |
| 90                   | 7.0                 |
| 59                   | 7.5                 |
| 66                   | 6.6                 |
| 87                   | 7.2                 |
| 90                   | 7.0                 |
| 106                  | 5.4                 |
| 94                   | 6.4                 |
| 57                   | 7.0                 |
| 138                  | 5.1                 |
| 87                   | 7.2                 |

```python
# 引入相关的包
import numpy as np
import pandas as pd 
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
# 加载数据
data=pd.read_csv("lm2.csv")
print(data)
# 准备数据
X = np.c_[data['x']]
y = np.c_[data['y']]
# 可视化数据
data.plot(kind="scatter", x="x", y="y")
plt.show()
# 选择一个模型进行训练
from sklearn import linear_model
lr_model =linear_model.LinearRegression()
lr_model.fit(X,y)
print("斜率：%s，截距：%s" %(lr_model.coef_[0][0], lr_model.intercept_[0]))
print("评估模型为：y=%sx + %sy"%(lr_model.coef_[0][0],lr_model.intercept_[0]))
# 画出拟合线
data.plot(kind="scatter",x="x", y="y")
plt.plot(X, lr_model.predict(X.reshape(-1,1)), color='red', linewidth=1)
plt.show()
# 对于新的问题进行预测
X_new=[[100]]
print(lr_model.predict(X_new))
```



