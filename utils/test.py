import numpy as np
from keras.models import load_model
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
from dataset.dataset import MyDataset
from utils.utils import cal_threshold, get_weights, draw_confusion_matrix, draw_roc_curve

def test_model(universal_config, test_config, model, X_thre, X_test, y_test):
    # Calculate threshold list
    e_threshold_list, e_mean, e_var = cal_threshold(model, X_thre, weights=get_weights(X_test, type='normal'))
    # self.threshold = e_mean + np.sqrt(e_var)
    threshold = np.percentile(e_threshold_list, universal_config["percent"])
    # threshold = 0.8093317163508755
    predictions_test = model.predict(X_test, batch_size=test_config["batch_size"])
    test_json = {"threshold": threshold}
    for weight_type in ['normal', 'linear', 'quadratic']:
        weights = get_weights(X_test, type=weight_type)
        e_test_list = []
        y_pred = []
        for i in range(predictions_test.shape[0]):
            x = predictions_test[i]
            y = X_test[i]
            e_i = np.sqrt(np.sum((x - y) * weights) ** 2)
            if e_i > threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
            e_test_list.append(e_i)
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        recall = recall_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred)
        f1 = f1_score(y_true=y_test, y_pred=y_pred)
        test_json[weight_type] = {
            'acc': acc,
            'recall': recall,
            'precision': precision,
            'f1': f1
        }
        print(f'测试集的Accuracy为:, {acc}, 权重类型为: {weight_type}')
        print(f'测试集的Recall为:, {recall}, 权重类型为: {weight_type}')
        print(f'测试集的Precision为:, {precision}, 权重类型为: {weight_type}')
        print(f'测试集的F1-score为:, {f1}, 权重类型为: {weight_type}')

        draw_confusion_matrix(y_test=y_test, y_pred=y_pred, exp_name=universal_config["exp_name"] + "_" + weight_type)
        draw_roc_curve(y_test=y_test, y_pred=e_test_list, exp_name=universal_config["exp_name"] + "_" + weight_type)

    with open(os.path.join("log", universal_config["exp_name"], "test_log.txt"), "w") as f:
        json.dump(test_json, f, indent=4)


'''
    Entrypoint for test
    Get the threshold and the test metrics
'''
def test(universal_config, test_config):
    anomaly_dataset = MyDataset(universal_config)
    x_train, x_threshold, x_test, y_test = anomaly_dataset.get_ad_data()
    print(f'Total len: {len(x_train) + len(x_threshold) + len(x_test)}\nx_train len: {len(x_train)}\nx_threshold len: {len(x_threshold)}\n'
        f'x_test len: {len(x_test)}, anomaly x_test len: {np.sum(y_test)}\nLoad data successfully!')

    vae = load_model(os.path.join('model_pth', universal_config["model_name"] + '.h5'))
    test_model(universal_config, test_config, model=vae, X_thre=x_threshold, X_test=x_test, y_test=y_test)
