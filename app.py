import pandas as pd
import numpy as np
from talib import BBANDS, SAR, RSI, STOCH, EMA, WILLR
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
# You can write code above the if-main block.

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # Read the training data
    train_df = pd.read_csv(args.training, names=("open", "high", "low", "close"))
    test_df = pd.read_csv(args.testing, names=("open", "high", "low", "close"))
    n_days_in = 15
    n_days_out = len(test_df)
    train_len = len(train_df)
    # Do MinMax normalization
    maxValue = train_df.to_numpy().max()
    minValue = train_df.to_numpy().min()
    diff = maxValue - minValue
    train = train_df.transform(lambda x: (x - minValue) / diff)
    test = test_df.transform(lambda x: (x - minValue) / diff)

    train = pd.concat([train, test], axis=0)
    train = train.reset_index(drop=True)
    
    # Expansion of data from data analysis
    train["upperband"], train["middleband"], train["lowerband"] = BBANDS(train.close.to_numpy())
    train["sar"] = SAR(train.high.to_numpy(), train.low.to_numpy())
    train["rsi"] = RSI(train.close.to_numpy(), timeperiod=5)
    train["slowk"], train["slowd"] = STOCH(train.high.to_numpy(), train.low.to_numpy(), train.close.to_numpy())
    train["ema"] = EMA(train.close.to_numpy(), timeperiod=5)
    train["willr"] = WILLR(train.high.to_numpy(), train.low.to_numpy(), train.close.to_numpy(), timeperiod=5)
    
    train_data = train.dropna() # WARNING: this cuts off the first few days
    train_data = train_data.reset_index(drop=True)
    train_len = len(train_data) - len(test_df)

    
    """
    test["upperband"], test["middleband"], test["lowerband"] = BBANDS(test.close.to_numpy())
    test["sar"] = SAR(test.high.to_numpy(), test.low.to_numpy())
    test["rsi"] = RSI(test.close.to_numpy(), timeperiod=5)
    test["slowk"], test["slowd"] = STOCH(test.high.to_numpy(), test.low.to_numpy(), test.close.to_numpy())
    """
    # 1->bullish, 0->bearish, 2->unknown
    y = list()
    tolerance = 0.1
    for i in range(train_len):
        # if train_data["open"][i+1] > train_data["open"][i] + tolerance:
        #     y.append(1)
        # elif train_data["open"][i+1] < train_data["open"][i] - tolerance:
        #     y.append(0)
        # else:
        #     y.append(0)
        if train_data["open"][i+1] > train_data["open"][i]:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y, dtype=np.int)

    # print("len(train_data): ", len(train_data))
    # print("len(train): ", len(train))
    # print("len(train_df): ", len(train_df))
    # print("len(y): ", len(y))
    # train_df = train_df[len(train)-len(train_data):].reset_index()
    # for i in range(len(y)):
    #     print(str(train_df["open"][i]) + " " + ("bull" if y[i] else "bear"))

    X = list()
    for i in range(n_days_in, len(train_data)):
        X.append(train_data.loc[i-n_days_in:i-1, :].values)
    X = np.array(X)
    
    y = y[n_days_in:]
   
    test = X[-n_days_out:]
    new_X = X[:-n_days_out]
    new_X = new_X.reshape((new_X.shape[0], -1))

    X_train, X_val, y_train, y_val = train_test_split(new_X, y, test_size=0.3, shuffle=False)
    # Use XGBClassifier and mclogloss to do multi-class classification
    xgb = XGBClassifier(learning_rate=0.1, 
                    objective='multi:softmax',
                    num_class=2,
                    n_estimators=30, max_depth=3, min_child_weight=10, use_label_encoder=False)
    
    model = xgb.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mlogloss",
                verbose=True)
    
    # Predict the testing data
    preds = model.predict(test.reshape(n_days_out, -1))
    ans = []
    unit = 0
    val = 0

    for i in range(1, len(preds)):
        # bullish
        if preds[i] == 1:
            if unit == 1:
                val = 0
            else:
                val = 1
                unit += 1
        # Do nothing
        # bearish
        else:
            if unit == -1:
                val = 0
            else:
                val = -1
                unit -= 1

        ans.append(val)


    balance = 0
    unit = 0
    hit = 0

    train_df = train_df[len(train)-len(train_data):].reset_index()
    # for i in range(len(y)):
    #     print(str(train_df["open"][i+n_days_in]) + " " + ("bull" if y[i] else "bear"))
    for i in range(len(test_df)-1):
        if (preds[i+1] == 1 and test_df["open"][i+1] > test_df["open"][i]) or (preds[i+1] == 0 and test_df["open"][i+1] <= test_df["open"][i]):
            hit += 1
        unit += ans[i]
        balance = balance - ans[i] * test_df["open"][i]
        if preds[i+1] == 1:
            status = "bull"
        elif preds[i+1] == 0:
            status = "bear"
        print(str(test_df["open"][i]) + " " + status + " " + str(ans[i]) + " " + str(unit) + " " + str(balance))
    print(test_df["open"][len(test_df)-1])
    if unit == 1:
        balance += test_df["close"][len(test_df)-1]
    elif unit == -1:
        balance -= test_df["close"][len(test_df)-1]

    print("Profit: ", balance)
    print("Accuracy: ", hit/19)
    
    # Write the result into output
    with open(args.output, "w") as fp:
        for i in range(len(ans)):
            print(ans[i], file=fp)