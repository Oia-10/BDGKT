import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score,r2_score
import numpy as np

detach = lambda o: o.cpu().detach().numpy().tolist()

def train(model, train_data, optimizer, device):
    model.train(mode=True)

    y_pred, y_true = [], []
    loss_all = []
    loss_func = nn.BCELoss()

    for batch_graph, tuser_id, titem_id, target_res, user, target_item, target_item_exist in train_data:
        
        optimizer.zero_grad()

        p  = model(batch_graph.to(device), 
                    tuser_id.to(device), 
                    titem_id.to(device), 
                    user.to(device), 
                    target_item.to(device),
                    target_item_exist.to(device),
                    )
        
        loss = loss_func(p, target_res.to(device).float())
        loss_all.append(loss.item())

        loss.backward()
        optimizer.step()

        y_pred += detach(p)
        y_true += detach(target_res.float())
                                

    fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
    mse_value = mean_squared_error(y_true, y_pred)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(y_true, y_pred)
    bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
    acc_value = accuracy_score(y_true, bi_y_pred)
    r2_value = r2_score(y_true, y_pred)

    return auc(fpr, tpr), np.mean(loss_all), mse_value, rmse_value, mae_value, acc_value, r2_value


def evaluate(model, val_data, device):
    model.eval()

    y_pred, y_true = [], []
    loss_all = []

    loss_func = nn.BCELoss()

    with torch.no_grad():
        for  batch_graph, tuser_id, titem_id, target_res, user, target_item, target_item_exist in val_data:
            p = model(batch_graph.to(device), 
                    tuser_id.to(device), 
                    titem_id.to(device), 
                    user.to(device), 
                    target_item.to(device),
                    target_item_exist.to(device),
                   )
            
            loss = loss_func(p, target_res.to(device).float())
            loss_all.append(loss.item())

            y_pred += detach(p)
            y_true += detach(target_res.float())
    
        fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
        mse_value = mean_squared_error(y_true, y_pred)
        rmse_value = np.sqrt(mse_value)
        mae_value = mean_absolute_error(y_true, y_pred)
        bi_y_pred = [1 if i >= 0.5 else 0 for i in y_pred]
        acc_value = accuracy_score(y_true, bi_y_pred)
        r2_value = r2_score(y_true, y_pred)
        
    return auc(fpr, tpr), np.mean(loss_all), mse_value, rmse_value, mae_value, acc_value, r2_value
