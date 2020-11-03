import pandas as pd


if __name__ == "__main__":
    model_list = ['CNN_LSTM_20201103-150358_0.3843.80', 'CNN_LSTM_20201103-160026_0.3879.75', 'CNN_LSTM_20201103-163042_0.3972.70', 'CNN_LSTM_20201103-174348_0.3880.80', 'CNN_LSTM_20201103-181044_0.3802.75', 'CNN_LSTM_20201103-181740_0.3850.75']
    folder = './predict/'
    predict = [0] * 227         # 227: test size
    for idx, model_name in enumerate(model_list):
        df = pd.read_csv(folder + model_name + '.csv').sort_values(['ID']).Label.to_list()
        predict = [a + b for a, b in zip(predict, df)]
    predict = [sum_value / len(model_list) for sum_value in predict]
    id = [_+1 for _ in range(len(predict))]

    ensemble_df = pd.DataFrame({'ID': id, 'Label': predict})
    ensemble_df.to_csv(folder + 'ensemble.csv', index=False)