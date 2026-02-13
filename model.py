import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class BDGKT(nn.Module):
    def __init__(self, student_num, question_num, skill_num, input_dim,
                 question_max_length, student_max_length,
                 drop1=0.2, drop2=0.2, layer_num=2, Q_KC=None):
        super(BDGKT, self).__init__()
        self.student_num = student_num
        self.question_num = question_num
        self.skill_num = skill_num
        self.question_max_length = question_max_length
        self.student_max_length = student_max_length
        self.hidden_size = input_dim
        self.layer_num = layer_num
        self.Q_KC = Q_KC

        self.student_static_embedding = nn.Embedding(self.student_num, self.hidden_size)
        self.question_static_embedding = nn.Embedding(self.question_num, self.hidden_size)
        self.skill_embedding = nn.Embedding(self.skill_num, self.hidden_size)
        self.response_embedding = nn.Embedding(2, self.hidden_size)

        self.layers = nn.ModuleList([
            BDGKTLayer(self.hidden_size, self.student_num, self.question_num,
                      self.student_max_length, self.question_max_length,
                      drop1, drop2)
            for _ in range(self.layer_num)
        ])

        self.student_layer_aggregator = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size)
        self.question_layer_aggregator = nn.Linear((self.layer_num + 1) * self.hidden_size, self.hidden_size)


        self.student_knowledge_pre = nn.Sequential(
            nn.Linear(self.hidden_size, skill_num),
            nn.Sigmoid())
        self.question_difficulty_pre= nn.Sequential(
            nn.Linear(self.hidden_size, skill_num),
            nn.Sigmoid())

        self.dynamic_state_fusion = nn.Linear(skill_num * 2, skill_num)
        self.question_difficulty_encoder = nn.Linear(self.hidden_size, skill_num)

        self.reset_parameters()

    def forward(self, graph, target_student_id, target_question_id,
                student_index, question_index, target_question_exist):
        device = target_question_id.device
        self.Q_KC = self.Q_KC.to(device)

        graph.nodes['student'].data['student_static'] = self.student_static_embedding(
            graph.nodes['student'].data['student_id'])
        graph.nodes['question'].data['question_static'] = self.question_static_embedding(
            graph.nodes['question'].data['question_id'])

        graph.nodes['student'].data['student_dynamic'] = graph.nodes['student'].data['student_static']
        graph.nodes['question'].data['question_dynamic'] = graph.nodes['question'].data['question_static']

        graph.edges['by'].data['response_h'] = self.response_embedding(
            graph.edges['by'].data['response'])
        graph.edges['pby'].data['response_h'] = self.response_embedding(
            graph.edges['pby'].data['response'])

        skill_multi = self.Q_KC[graph.nodes['question'].data['question_id']]
        graph.nodes['question'].data['skill'] = torch.matmul(
            skill_multi, self.skill_embedding.weight) / torch.sum(skill_multi, dim=-1).unsqueeze(-1)

        target_question_static_embedding = self.question_static_embedding(target_question_id)
        target_student_static_embedding = self.student_static_embedding(target_student_id)

        node_features = None
        student_layers = []
        question_layers = []

        student_layers.append(
            extract_student_features(graph, student_index, graph.nodes['student'].data['student_dynamic']))
        question_layers.append(
            extract_question_features(graph, question_index, graph.nodes['question'].data['question_dynamic']))

        if self.layer_num > 0:
            for layer in self.layers:
                node_features = layer(graph, node_features)

                student = extract_student_features(graph, student_index, node_features['student'])
                student_layers.append(student)

                question = extract_question_features(graph, question_index, node_features['question'])
                question = torch.where(
                    target_question_exist.unsqueeze(-1).bool(),
                    question,
                    target_question_static_embedding)
                question_layers.append(question)


        target_student_dynamic_embedding = self.student_layer_aggregator(torch.cat(student_layers, -1))
        target_question_dynamic_embedding = self.question_layer_aggregator(torch.cat(question_layers, -1))

        question_difficulty = self.question_difficulty_pre(target_question_dynamic_embedding)
        student_knowledge = self.student_knowledge_pre(target_student_dynamic_embedding)

        dynamic_factor = torch.sigmoid(
            self.dynamic_state_fusion(torch.cat([student_knowledge, (1 - question_difficulty)], dim=-1)))
        target_abs_qdiff = torch.sigmoid(self.question_difficulty_encoder(target_question_static_embedding))

        exp = torch.exp(-1.702 * 5 * (dynamic_factor - target_abs_qdiff))
        probability = 1 / (1 + exp)

        target_skill_multi = self.Q_KC[target_question_id]
        pred = torch.sum(probability * target_skill_multi, dim=-1) / torch.sum(target_skill_multi, dim=-1)

        return pred

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)


class BDGKTLayer(nn.Module):
    def __init__(self, hidden_size, student_num, question_num,
                 student_max_length, question_max_length, drop1, drop2):
        super(BDGKTLayer, self).__init__()
        self.hidden_size = hidden_size
        self.student_num = student_num
        self.question_num = question_num
        self.student_max_length = student_max_length
        self.question_max_length = question_max_length
        self.feature_dropout = nn.Dropout(drop1)
        self.attention_dropout = nn.Dropout(drop2)

        self.student_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.question_weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.knowledge_init = nn.Parameter(torch.rand(1, self.hidden_size, dtype=torch.float32))
        self.question_transform = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.response_output = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.question_query = nn.Linear(self.hidden_size, self.hidden_size)
        self.forget_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.question_abs_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attention_key = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.attention_query = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_value = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, graph, node_features=None):
        if node_features is None:
            student_feat = graph.nodes['student'].data['student_dynamic']
            question_feat = graph.nodes['question'].data['question_dynamic']
        else:
            student_feat = node_features['student']
            question_feat = node_features['question']

        graph.nodes['student'].data['student_dynamic'] = self.student_weight(
            self.feature_dropout(student_feat))
        graph.nodes['question'].data['question_dynamic'] = self.question_weight(
            self.feature_dropout(question_feat))

        graph = self.graph_update(graph)

        updated_features = {
            'student': graph.nodes['student'].data['student_dynamic'],
            'question': graph.nodes['question'].data['question_dynamic']
        }

        return updated_features

    def graph_update(self, graph):
        graph.multi_update_all(
            {
                'by': (self.student_message_func, self.student_reduce_func),
                'pby': (self.question_message_func, self.question_reduce_func)
            },
            'sum'
        )
        return graph

    def question_message_func(self, edges):
        return {
            'student_dynamic': edges.src['student_dynamic'],
            'student_static': edges.src['student_static'],
            'question_dynamic': edges.dst['question_dynamic'],
            'question_static': edges.dst['question_static'],
            'skill': edges.dst['skill'],
            'response_h': edges.data['response_h'],
            'timestamp': edges.data['timestamp']
        }

    def question_reduce_func(self, nodes):
        question_abs = self.question_abs_layer(torch.cat([
            nodes.mailbox['question_static'],
            nodes.mailbox['skill']
        ], dim=-1))

        key = self.attention_key(torch.cat([
            nodes.mailbox['student_dynamic'],
            question_abs,
            nodes.mailbox['response_h']
        ], dim=-1))

        query = self.attention_query(question_abs)
        value = self.attention_value(torch.cat([
            question_abs,
            nodes.mailbox['response_h']
        ], dim=-1))

        e_ij = torch.sum(query * key, dim=2) / torch.sqrt(torch.tensor(self.hidden_size).float())
        alpha = self.attention_dropout(F.softmax(e_ij, dim=1))

        if len(alpha.shape) == 2:
            alpha = alpha.unsqueeze(2)

        h = torch.sum(alpha * value, dim=1)

        return {'question_dynamic': h}

    def student_message_func(self, edges):
        return {
            'question_dynamic': edges.src['question_dynamic'],
            'question_static': edges.src['question_static'],
            'skill': edges.src['skill'],
            'student_dynamic': edges.dst['student_dynamic'],
            'student_static': edges.dst['student_static'],
            'response_h': edges.data['response_h'],
            'timestamp': edges.data['timestamp']
        }

    def student_reduce_func(self, nodes):
        question_abs = self.question_abs_layer(torch.cat([
            nodes.mailbox['question_static'],
            nodes.mailbox['skill']
        ], dim=-1))

        knowledge_state = torch.tile(
            self.knowledge_init,
            (nodes.mailbox['student_dynamic'].size(0), 1) )


        for i in range(nodes.mailbox['response_h'].size(1)):
            q_transform = self.question_transform(torch.cat((
                nodes.mailbox['question_dynamic'][:, i, :],
                question_abs[:, i, :],
                knowledge_state
            ), dim=-1))

            q_activated = torch.tanh(self.question_query(q_transform))
            x = torch.cat([q_transform, nodes.mailbox['response_h'][:, i, :]], dim=-1)
            response_out = torch.sigmoid(self.response_output(x))
            response_out = response_out * q_activated

            forget_gate_val = torch.sigmoid(self.forget_gate(torch.cat([
                nodes.mailbox['response_h'][:, i, :],
                knowledge_state
            ], dim=-1)))

            knowledge_state = forget_gate_val * knowledge_state + (1 - forget_gate_val) * response_out

        return {'student_dynamic': knowledge_state}


def extract_student_features(batch_graph, student_index, student_embedding):
    batch_student_size = batch_graph.batch_num_nodes('student')
    cumsum = torch.cumsum(batch_student_size, 0)
    offset = torch.roll(cumsum, 1)
    offset[0] = 0
    new_student_index = offset + student_index
    return student_embedding[new_student_index]


def extract_question_features(batch_graph, question_index, question_embedding):
    batch_question_size = batch_graph.batch_num_nodes('question')
    cumsum = torch.cumsum(batch_question_size, 0)
    offset = torch.roll(cumsum, 1)
    offset[0] = 0
    new_question_index = offset + question_index
    return question_embedding[new_question_index]
