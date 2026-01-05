import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

class BDGKT(nn.Module):
    def __init__(self, user_num, item_num, skill_num, input_dim, 
                 item_max_length, user_max_length, 
                 p_drop=0.2,feat_drop = 0.2, attn_drop = 0.2, layer_num = 2, 
                 Q_KC = None):
        super(BDGKT, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.skill_num = skill_num
        self.item_max_length = item_max_length
        self.user_max_length = user_max_length
        self.hidden_size = input_dim
        self.layer_num = layer_num

        self.Q_KC = Q_KC

        self.u_static_embedding = nn.Embedding(self.user_num, self.hidden_size) 
        self.q_static_embedding = nn.Embedding(self.item_num, self.hidden_size) 
        self.s_embedding = nn.Embedding(self.skill_num, self.hidden_size) 
        self.r_embedding = nn.Embedding(2, self.hidden_size) 

        self.layers = nn.ModuleList([BDGKTLayers(self.hidden_size, self.hidden_size, 
                                                self.user_num, self.item_num, 
                                                self.user_max_length, self.item_max_length, 
                                                feat_drop, attn_drop) for _ in range(self.layer_num)])

        self.unified_user = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size)
        self.unified_item = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size)

        self.trans_dynamic_user = nn.Sequential(
            nn.Linear(self.hidden_size, skill_num), 
            nn.Sigmoid(),
        )
        self.trans_dynamic_item = nn.Sequential(
            nn.Linear(self.hidden_size, skill_num), 
            nn.Sigmoid(),
        )

        self.trans_dynamic_state = nn.Linear(skill_num * 2, skill_num)
        self.trans_abs_diff = nn.Linear(self.hidden_size,skill_num) 
       
        self.reset_parameters()


    def forward(self, g, tuser_id, titem_id, user_index, item_index, target_item_exist):
        device = titem_id.device

        self.Q_KC = self.Q_KC.to(device) 

        g.nodes['user'].data['user_static'] = self.u_static_embedding(g.nodes['user'].data['user_id']) 
        g.nodes['item'].data['item_static'] = self.q_static_embedding(g.nodes['item'].data['item_id']) 

        g.nodes['user'].data['user_dynamic'] = g.nodes['user'].data['user_static']
        g.nodes['item'].data['item_dynamic'] = g.nodes['item'].data['item_static'] 

        g.edges['by'].data['response_h'] = self.r_embedding(g.edges['by'].data['response']) 
        g.edges['pby'].data['response_h'] = self.r_embedding(g.edges['pby'].data['response']) 
       
        skill_multi = self.Q_KC[g.nodes['item'].data['item_id']]
        g.nodes['item'].data['skill'] = torch.matmul(skill_multi, self.s_embedding.weight) / torch.sum(skill_multi, dim=-1).unsqueeze(-1) 

        target_item_static_embedding = self.q_static_embedding(titem_id) 
        target_user_static_embedding = self.u_static_embedding(tuser_id) 
        
        feat_dict = None
        user_layer = []
        item_layer = []

        user_layer.append(graph_user(g, user_index, g.nodes['user'].data['user_dynamic']))
        item_layer.append(graph_item(g, item_index, g.nodes['item'].data['item_dynamic']))

        if self.layer_num > 0:
            for conv in self.layers:
                feat_dict = conv(g, feat_dict)

                user = graph_user(g, user_index, feat_dict['user'])
                user_layer.append(user)

                item = graph_item(g, item_index, feat_dict['item'])
                item = torch.where(target_item_exist.unsqueeze(-1).bool(), item, target_item_static_embedding)
                item_layer.append(item)

        target_user_dynamic_embedding = self.unified_user(torch.cat(user_layer, -1)) 
        target_item_dynamic_embedding = self.unified_item(torch.cat(item_layer, -1)) 


        RD = self.trans_dynamic_item(target_item_dynamic_embedding)
        user_01 = self.trans_dynamic_user(target_user_dynamic_embedding)

        DF = torch.sigmoid(self.trans_dynamic_state(torch.cat([user_01, (1-RD)],dim=-1)))
        target_abs_qdiff = torch.sigmoid(self.trans_abs_diff(target_item_static_embedding)) 

        exp = torch.exp(5 * (DF -  target_abs_qdiff)) 
        P =  1 / (1 + exp)

        target_skill_multi = self.Q_KC[titem_id]
        pred = torch.sum(P * target_skill_multi, dim=-1) / torch.sum(target_skill_multi, dim=-1)

        return pred

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)


class BDGKTLayers(nn.Module):
    def __init__(self, in_feats, out_feats, 
                 user_num, item_num, 
                 user_max_length, item_max_length, 
                 feat_drop, attn_drop):
        super(BDGKTLayers, self).__init__()
        self.hidden_size = in_feats
        self.user_num = user_num
        self.item_num = item_num
        self.user_max_length = user_max_length
        self.item_max_length = item_max_length
        self.feat_drop = nn.Dropout(feat_drop)
        self.atten_drop = nn.Dropout(attn_drop)

        self.user_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.item_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.knowledge_init = nn.Parameter(torch.rand(1, self.hidden_size, dtype=torch.float32))
        self.q1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.Lo = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.Lq = nn.Linear(self.hidden_size, self.hidden_size)
        self.fo_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.l1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l4 = nn.Linear(self.hidden_size * 2, self.hidden_size)
    

    def forward(self, g, feat_dict=None):
        if feat_dict == None:
            user_ = g.nodes['user'].data['user_dynamic']
            item_ = g.nodes['item'].data['item_dynamic']
        else:
            user_ = feat_dict['user']
            item_ = feat_dict['item']

        g.nodes['user'].data['user_dynamic'] = self.user_weight(self.feat_drop(user_))
        g.nodes['item'].data['item_dynamic'] = self.item_weight(self.feat_drop(item_))
        
        g = self.graph_update(g)

        f_dict = {'user': g.nodes['user'].data['user_dynamic'], 'item': g.nodes['item'].data['item_dynamic']}

        return f_dict

    def graph_update(self, g):
        g.multi_update_all({'by': (self.user_message_func, self.user_reduce_func),
                            'pby': (self.item_message_func, self.item_reduce_func)}, 'sum')
        return g

    def item_message_func(self, edges):
        dic = {}
        dic['user_dynamic'] = edges.src['user_dynamic']
        dic['user_static'] = edges.src['user_static']
        dic['item_dynamic'] = edges.dst['item_dynamic']
        dic['item_static'] = edges.dst['item_static']
        dic['skill'] = edges.dst['skill']
        dic['response_h'] = edges.data['response_h']
        dic['timestamp'] = edges.data['timestamp']

        return dic

    def item_reduce_func(self, nodes):
        h = []

        item_abs = self.l1(torch.cat([nodes.mailbox['item_static'],nodes.mailbox['skill']], dim=-1))

        key = self.l2(torch.cat([nodes.mailbox['user_dynamic'],item_abs,nodes.mailbox['response_h']],dim=-1))
        Query = self.l3(item_abs)
        Value = self.l4(torch.cat([item_abs,nodes.mailbox['response_h']],dim=-1))


        e_ij = torch.sum(Query * key, dim=2)/torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.atten_drop(F.softmax(e_ij, dim=1))
        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)
        h_long = torch.sum(alpha * Value, dim=1)
        h.append(h_long)

        return {'item_dynamic': h[0]}
        

    def user_message_func(self, edges):
        dic = {}
        dic['item_dynamic'] = edges.src['item_dynamic']
        dic['item_static'] = edges.src['item_static']
        dic['skill'] = edges.src['skill']
        dic['user_dynamic'] =  edges.dst['user_dynamic']
        dic['user_static'] =  edges.dst['user_static']
        dic['response_h'] = edges.data['response_h']
        dic['timestamp'] = edges.data['timestamp']

        return dic

    def user_reduce_func(self, nodes):
        h = []
        item_abs = self.l1(torch.cat([nodes.mailbox['item_static'],nodes.mailbox['skill']], dim=-1))
        
        kkk_step =  torch.tile(self.knowledge_init, (nodes.mailbox['user_dynamic'].size(0), 1))

        for i in range(0, nodes.mailbox['response_h'].size(1)): 
            q1 = self.q1(torch.cat((nodes.mailbox['item_dynamic'][:, i, :],
                                    item_abs[:, i, :],  
                                    kkk_step), dim=-1))

            xx_title = torch.tanh(self.Lq(q1))
            x = torch.cat([q1, nodes.mailbox['response_h'][:, i, :]],dim =-1)
            xx = torch.sigmoid(self.Lo(x))
            xx = xx * xx_title

            foin = torch.sigmoid(self.fo_gate(torch.cat([
                                                         nodes.mailbox['response_h'][:, i, :],
                                                         kkk_step], dim=-1)))
            kkk_step = foin * kkk_step + (1 - foin) * xx
        h.append(kkk_step)

        return {'user_dynamic': h[0]}
       

def graph_user(bg, user_index, user_embedding):
    b_user_size = bg.batch_num_nodes('user')
    tmp = torch.roll(torch.cumsum(b_user_size, 0), 1)
    tmp[0] = 0
    new_user_index = tmp + user_index
    return user_embedding[new_user_index]

def graph_item(bg, last_index, item_embedding):
    b_item_size = bg.batch_num_nodes('item')
    tmp = torch.roll(torch.cumsum(b_item_size, 0), 1)
    tmp[0] = 0
    new_item_index = tmp + last_index
    return item_embedding[new_item_index]


