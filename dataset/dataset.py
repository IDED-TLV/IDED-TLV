# coding=utf-8
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold


class MyDataset:
    def __init__(self, universal_config):
        self.universal_config = universal_config
        self.time_step = self.universal_config['time_step']
        self.data_path = os.path.join("data", self.universal_config['data_name'] + '.xlsx')
        self.used_sheet_names = pd.ExcelFile(self.data_path).sheet_names
        self.test_sheet_names = ['16', '17']
        self.train_sheet_names = [name for name in self.used_sheet_names if name not in self.test_sheet_names]

        self.scaler = StandardScaler()
        self.need_fitting = True
        self.params = ['压差ΔP', '出口温度T22', '体积流量Q', '入口温度T3', '水合物浆液密度', '水转化率', '数据标记']

        self.df_info = {}
        for sheet_name in self.used_sheet_names:
            df = pd.read_excel(self.data_path, sheet_name=sheet_name, header=None, usecols="A:H", skiprows=2)
            df = self._preprocess_for_df(df)
            self.df_info[sheet_name] = {}
            self.df_info[sheet_name]['data'] = df
            self.df_info[sheet_name]['len'] = len(df)
            self.df_info[sheet_name]['len_correct'] = len(df.loc[df["数据标记"] == 0])

        # 用于累加 len 和 len_correct 的值
        total_len = sum(item["len"] for item in self.df_info.values())
        total_len_correct = sum(item["len_correct"] for item in self.df_info.values())

        print("Total len:", total_len)
        print("Total len_correct:", total_len_correct)

        print(f'数据读取完毕！')

    def _transform(self, df):
        flag = 0
        if df['数据标记'] != 0:
            flag = 1
        return flag

    def _preprocess_for_df(self, df):
        # 简单处理每个df
        df.columns = ['时间t', '压差ΔP', '出口温度T22',
                      '入口温度T3', '水合物浆液密度', '体积流量Q', '水转化率', '数据标记']
        df = df.dropna(
            subset=['时间t', '压差ΔP', '出口温度T22', '入口温度T3', '体积流量Q',
                    '数据标记']).loc[:, self.params]

        df.loc[:, "数据标记"] = df.apply(self._transform, axis=1)
        return df

    def get_ad_data(self):
        total_x = []
        total_y = []

        for sheet_name in self.train_sheet_names:
            df = self.df_info[sheet_name]['data']
            for i in range(len(df)):
                total_x.append(df.iloc[i, :-1])
                total_y.append(df.iloc[i, -1])

        self.scaler.fit(total_x)
        joblib.dump(self.scaler, os.path.join("log", self.universal_config['exp_name'], "scaler.joblib"))
        print(f"Scaler was saved in {os.path.join("log", self.universal_config['exp_name'], "scaler.joblib")}！")
        total_x = self.scaler.transform(total_x)

        # 归一化后，分开处理每一个df
        cnt = 0
        x_correct_series = []
        x_anomaly_series = []
        for sheet_idx, sheet_name in enumerate(self.train_sheet_names):
            data_len = self.df_info[sheet_name]['len']
            data_len_correct = self.df_info[sheet_name]['len_correct']
            for i in range(cnt, cnt + data_len - self.time_step):
                if i < cnt + data_len_correct - self.time_step:
                    x_correct_series.append(total_x[i:i + self.time_step])
                else:
                    x_anomaly_series.append(total_x[i:i + self.time_step])
            cnt += data_len

        x_correct = np.array(x_correct_series)
        x_anomaly = np.array(x_anomaly_series)

        indices = np.random.permutation(len(x_correct))
        x_correct = x_correct[indices]

        # 正确数据82分为两份，一份用于训练模型，一份用于确定阈值
        x_train = x_correct[: int(0.8 * len(x_correct))]
        x_valid = x_correct[int(0.8 * len(x_correct)):]

        # 构建测试集
        df_test = pd.DataFrame(columns=self.params)
        for sheet_name in self.test_sheet_names:
            df_test = pd.concat([df_test, self.df_info[sheet_name]['data']])

        x_test = df_test.iloc[:, :-1]
        x_test = self.scaler.transform(x_test.values)
        y_test = df_test.iloc[:, -1].values
        x_test_series = []
        y_test_series = []
        cnt = 0
        for sheet_idx, sheet_name in enumerate(self.test_sheet_names):
            data_len = self.df_info[sheet_name]['len']
            normal_cnt = 0
            anomaly_cnt = 0
            for i in range(cnt, cnt + data_len - self.time_step):
                x_test_series.append(x_test[i: i + self.time_step])
                if y_test[i + self.time_step] == 0:
                    normal_cnt += 1
                else:
                    anomaly_cnt += 1
                y_test_series.append(y_test[i + self.time_step])
            print('正常数据: ', normal_cnt, ' 异常数据: ', anomaly_cnt)
            cnt += data_len
        print('其他数据的异常数据数量: ', len(x_anomaly))
        # 测试集的组成：选定的file加上训练集、验证集对应的异常数据
        x_test = np.array(x_test_series)
        x_test = np.concatenate([x_test, x_anomaly])
        y_test = np.array(y_test_series)
        y_test = np.concatenate([y_test, np.ones(len(x_anomaly))])
        print('测试集数据数量：', len(x_test))
        print('测试集异常数据数量：', sum(y_test))
        return x_train, x_valid, x_test, y_test


if __name__ == "__main__":
    with open(os.path.join("config", "config.json"), "r") as f:
        config = json.load(f)
    universal_config, train_config, test_config = config["universal"], config["train"], config["test"]
    anomaly_dataset = MyDataset(universal_config)
    x_train, x_threshold, x_test, y_test = anomaly_dataset.get_ad_data()
    print(
        f'Total len: {len(x_train) + len(x_threshold) + len(x_test)}\nx_train len: {len(x_train)}\nx_threshold len: {len(x_threshold)}\n'
        f'x_test len: {len(x_test)}, anomaly x_test len: {np.sum(y_test)}\nLoad data successfully!')
