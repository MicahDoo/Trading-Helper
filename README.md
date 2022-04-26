# Auto Trading
In this project, we will implement a very aged prediction problem from the financial field.From a series of stock prices, including daily open, high, low, and close prices, decide our daily action and make our best profit for the future trading.
## Model Algorithm
>XGBoost works as Newton-Raphson in function space unlike gradient boosting that works as gradient descent in function space, a second order Taylor approximation is used in the loss function to make the connection to Newton Raphson method. [詳細內容可以再參照 (Tianqi Chen&Carlos Guestrin,2016)](https://arxiv.org/pdf/1603.02754.pdf)

## Features we feed
我們這次使用 XGBoost 演算法，
將股票**前20天的股市 opening price 當作 features**，
feed 給 XGBoost 演算法當作已知的 feature (X)，
再嘗試 predict 出 表示**當日與明日股票的升降狀況 y label**。
## Predict way
1. 先預測股票升降
    - 股價升：得到的 `predict y == 1`
    - 股價降：得到的 `predict y == 0`
2. 再根據升降的預測結果和當前所持有的股票數判斷
    | 明天股價變化 | 目前持有股票數 | 行動 |
    | -------- | -------- | -------- |
    | 升     | 1     | 持平     |
    | 降     | 1     | 賣出     |
    | 升     | 0     | 買入     |
    | 降     | 0     | 賣出     |
    | 升     | -1     | 買入     |
    | 降     | -1     | 持平     |
## Result 

| Open | Prediction | Action | Shares | Balance    |
| ---- | ---------- | ------ | ------ | --- |
154.4 |bull| 1| 1| -154.4
155.96| bear| -1| 0| 1.5600000000000023
156.45| bear| -1| -1| 158.01
154.1| bull| 1| 0| 3.9099999999999966
153.59| bull| 1| 1| -149.68
154.81 |bull| 0| 1| -149.68
155.46| bull| 0| 1| -149.68
156.74| bear| -1| 0| 7.060000000000002
156.6| bull| 1| 1| -149.54
154.6| bull| 0| 1| -149.54
153.61| bull| 0| 1| -149.54
153.59|bull| 0| 1| -149.54
154.05| bull| 0| 1| -149.54
153.65| bear| -1| 0| 4.110000000000014
153.17| bull| 1| 1| -149.05999999999997
151.82| bull| 0| 1| -149.05999999999997
152.51| bull| 0| 1| -149.05999999999997
152.95| bull| 0| 1| -149.05999999999997
153.2| bull| 0| 1| -149.05999999999997
154.17
## Performance
```
Profit:  4.360000000000014
Accuracy:  0.631578947368421
```


