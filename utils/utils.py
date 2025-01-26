import numpy as np
import keras
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, \
    roc_curve, auc, ConfusionMatrixDisplay
from keras.models import load_model
import os
from dataset.dataset import MyDataset
import matplotlib.pyplot as plt
import pandas as pd


# 根据type返回不同类型的权重
def get_weights(X_test, type='normal'):
    time_steps = X_test.shape[1]
    feature_dim = X_test.shape[2]
    if type == 'normal':
        weights_for_normal = [1.0 / time_steps for _ in range(time_steps)]
        weights_for_normal = np.repeat(weights_for_normal, feature_dim).reshape(time_steps, feature_dim)
        return weights_for_normal
    elif type == 'linear':
        weights_for_ascent = np.linspace(1, 10, time_steps)  # 权重从 1 增长到 10
        # 归一化权重，使它们的和为 1
        weights_for_ascent /= np.sum(weights_for_ascent)
        weights_for_ascent = np.repeat(weights_for_ascent, feature_dim).reshape(time_steps, feature_dim)
        return weights_for_ascent
    elif type == 'quadratic':
        # 生成二次函数权重
        weights_quadratic = np.linspace(0, 10, time_steps) ** 2
        # 归一化权重
        weights_quadratic /= np.sum(weights_quadratic)
        weights_quadratic = np.repeat(weights_quadratic, feature_dim).reshape(time_steps, feature_dim)
        return weights_quadratic
    elif type == 'exp':
        pass


def cal_threshold(model, X_thre, weights):
    e_threshold_list = []
    predictions_train = model.predict(X_thre, batch_size=128)
    for i in range(predictions_train.shape[0]):
        x = predictions_train[i]
        y = X_thre[i]
        e_threshold_list.append(np.sqrt(np.sum(((x - y) * weights) ** 2)))
    e_mean = np.average(e_threshold_list)
    e_var = np.var(e_threshold_list)
    return e_threshold_list, e_mean, e_var


def adjust_result(model, batch_size, x_order, y_order, cnt_threshold, err_threshold):
    print('调整实验结果！')
    y_true_all = []
    y_pred_adjusted_all = []
    err_list = []
    for filename, x in x_order.items():
        x_pred = model.predict(x, batch_size=batch_size)
        exc_cnt = 0
        turning_point_pred = 0
        y_pred = []
        for i in range(x_pred.shape[0]):
            pred_i = x_pred[i].flatten()
            true_i = x[i].flatten()
            e_i = np.sqrt(np.sum((pred_i - true_i) ** 2))
            if e_i > err_threshold:
                exc_cnt += 1
                y_pred.append(1)
            else:
                exc_cnt = 0
                y_pred.append(0)
            if exc_cnt >= cnt_threshold:
                turning_point_pred = i - exc_cnt
                break
        y_true = y_order[filename]
        turning_point_true = np.where(y_true == 1)[0][0] if len(np.where(y_true == 1)[0]) > 0 else 0
        err_list.append(abs(turning_point_true - turning_point_pred) / len(y_true))
        y_pred_adjusted = np.zeros(len(y_true), dtype=int)
        y_pred_adjusted[turning_point_pred:] = 1
        y_pred_adjusted_all.extend(y_pred_adjusted)
        y_true_all.extend(y_true)
    print('转折点的预测误差为:', sum(err_list) / len(x_order))  # 用每个df的误差加起来除以文件数量
    print("调整后的的Accuracy为:", accuracy_score(y_true=y_true_all, y_pred=y_pred_adjusted_all))
    print("测试集的Recall为:", recall_score(y_true=y_true_all, y_pred=y_pred_adjusted_all))
    print("测试集的Precision为:", precision_score(y_true=y_true_all, y_pred=y_pred_adjusted_all))
    print("测试集的F1-score为:", f1_score(y_true=y_true_all, y_pred=y_pred_adjusted_all))


def visualize(result_path, model, vis_origin_data, vis_series_data, y, file_no, threshold):
    print('可视化数据！')
    # 找到真实数据中的第一个1
    if len(np.where(y == 1)[0]) > 0:
        first_1 = np.where(y == 1)[0][0]
    else:
        first_1 = -1

    if not os.path.exists(os.path.join(result_path, file_no)):
        os.mkdir(os.path.join(result_path, file_no))
    pred = model.predict(vis_series_data)
    # 取最后一个时间点，第一个特征（压差）
    pressure_pred = pred[:, -1, 0]
    plt.figure(figsize=(12, 10))
    plt.plot(vis_origin_data[:, 0], label='true')
    plt.legend()
    plt.ylabel('delta P')
    plt.savefig(os.path.join(result_path, file_no, f'{file_no} 原始压差.png'))
    plt.clf()

    plt.figure(figsize=(12, 10))
    plt.plot(vis_origin_data[:, 0], label='true')
    plt.plot(pressure_pred, label='pred')
    if first_1 > -1:
        plt.axvline(x=first_1, linestyle='--', color='r')
    # plt.title(f'{file_no}. Change of delta P')
    plt.legend()
    plt.ylabel('delta P')
    plt.savefig(os.path.join(result_path, file_no, f'{file_no} 压差.png'))
    plt.clf()
    # 取最后一个时间点，第2个特征（出口温度）
    pressure_pred = pred[:, -1, 1]
    plt.figure(figsize=(12, 10))
    plt.plot(vis_origin_data[:, 1], label='true')
    plt.plot(pressure_pred, label='pred')
    if first_1 > -1:
        plt.axvline(x=first_1, linestyle='--', color='r')
    # plt.title(f'{file_no}. Change of output T')
    plt.legend()
    plt.ylabel('T')
    plt.savefig(os.path.join(result_path, file_no, f'{file_no} 出口温度.png'))
    plt.clf()
    # 取最后一个时间点，第3个特征（体积流量）
    pressure_pred = pred[:, -1, 2]
    plt.figure(figsize=(12, 10))
    plt.plot(vis_origin_data[:, 2], label='true')
    plt.plot(pressure_pred, label='pred')
    if first_1 > -1:
        plt.axvline(x=first_1, linestyle='--', color='r')
    # plt.title(f'{file_no}. Change of Q')
    plt.legend()
    plt.ylabel('Q')
    plt.savefig(os.path.join(result_path, file_no, f'{file_no} 体积流量.png'))
    plt.clf()

    error_list = []
    y_pred = []
    for i in range(pred.shape[0]):
        x = pred[i].flatten()
        y = vis_series_data[i].flatten()
        e_i = np.sqrt(np.sum((x - y) ** 2))
        error_list.append(e_i)
        if e_i > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    # print(y_pred)
    plt.figure(figsize=(16, 14))
    plt.plot(error_list)
    plt.title('Change of error')
    plt.axhline(y=threshold, linestyle='--', color='r')
    if first_1 > -1:
        plt.axvline(x=first_1, linestyle='--', color='r')
    plt.ylabel('error')
    plt.savefig(os.path.join(result_path, file_no, f'{file_no} 误差.png'))
    plt.clf()


def draw_confusion_matrix(y_test, y_pred, exp_name):
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join("log", exp_name, "confusion_matrix.png"))


def draw_roc_curve(y_test, y_pred, exp_name):
    # 计算ROC曲线和AUC值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # 输出到文件
    df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})

    # 将DataFrame保存到CSV文件
    df.to_csv(os.path.join('log', exp_name, 'roc_curve.csv'),
              index=False)

    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("log", exp_name, "roc_curve.png"))
