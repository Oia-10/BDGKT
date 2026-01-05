import time
import torch
import torch.optim as optim
import datetime
import copy
from torch.utils.data import DataLoader
import os
from dgl import load_graphs
from config import *
from utils import *
from dataload import *
from trainUtils import *
from model import *

opt = set_opt()
print(opt)

set_seed(opt.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_root = f'./graphdata/{opt.dataset}/{opt.cv_num}/{opt.dataset}_{opt.item_max_length}_{opt.user_max_length}/train/'
val_root = f'./graphdata/{opt.dataset}/{opt.cv_num}/{opt.dataset}_{opt.item_max_length}_{opt.user_max_length}/val/'
test_root = f'./graphdata/{opt.dataset}/{opt.cv_num}/{opt.dataset}_{opt.item_max_length}_{opt.user_max_length}/test/'

train_set = myFloder(train_root, load_graphs)
test_set = myFloder(test_root, load_graphs)
val_set = myFloder(val_root, load_graphs)

train_data = DataLoader(dataset = train_set, 
                        batch_size = opt.batch_size, 
                        collate_fn = collate, 
                        shuffle = True, 
                        pin_memory = True, 
                        num_workers = 12)
test_data = DataLoader(dataset = test_set,
                       batch_size = opt.batch_size,
                       collate_fn = collate,
                       pin_memory = True,
                       num_workers = 8)
val_data = DataLoader(dataset = val_set,
                      batch_size = opt.batch_size,
                      collate_fn = collate,
                      pin_memory = True,
                      num_workers = 2)

Q_KC_path =  './data/' + opt.dataset+'/' + 'Q_KC.pt'
Q_KC = torch.load(Q_KC_path)

model = BDGKT(user_num = opt.user_num, item_num = opt.item_num, skill_num = opt.skill_num, input_dim=opt.hidden_size, 
             item_max_length=opt.item_max_length, user_max_length=opt.user_max_length, 
             feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, p_drop=opt.p_drop,
             layer_num=opt.layer_num, Q_KC = Q_KC).to(device)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
early_stopping = EarlyStopping(patience=opt.patience, verbose=True) 

result_path =  './result/'+ opt.dataset + '/%s' % ('{0:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()))

os.makedirs(result_path)
info_file = open('%s/info.txt' % result_path, 'w+')

save_info_opt(opt.dataset, opt.batch_size, opt.hidden_size, opt.lr, opt.l2, opt.feat_drop, opt.attn_drop,  opt.p_drop,
            opt.layer_num, opt.item_max_length, opt.user_max_length, opt.patience, opt.gpu, opt.cv_num, opt.seed, 
            info_file)


max_auc = 0.0
for epoch in range(1, opt.epoch + 1):
    train_auc, train_loss, train_mse, train_rmse, train_mae, train_acc, train_r2 = train(model, train_data, optimizer, device)
    valid_auc, valid_loss, valid_mse, valid_rmse, valid_mae, valid_acc, valid_r2 = evaluate(model, val_data, device)

    if max_auc < valid_auc:
        max_auc = valid_auc
        torch.save(model.state_dict(), '%s/model' % ('%s' % result_path))
        current_max_model = copy.deepcopy(model)
    
    save_info_train_valid(opt.cv_num, epoch, max_auc,
                valid_auc, valid_loss, valid_mse, valid_rmse,valid_mae, valid_acc, valid_r2,
                train_auc, train_loss, train_mse, train_rmse,train_mae, train_acc, train_r2,
                info_file)

    early_stopping(valid_loss)    
    if early_stopping.early_stop:
        print("Early stopping")
      

        print('The training has been completed and the final result is: ')
        info_file.write('The training has been completed and the final result is: ')
        test_auc, test_loss, test_mse, test_rmse, test_mae, test_acc, test_r2 = evaluate(current_max_model, test_data, device)
        save_info_test(opt.cv_num, test_auc, test_loss, test_mse, test_rmse, test_mae, test_acc,  test_r2, info_file)

        break

    

