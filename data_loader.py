import os
from torch.utils.data import Dataset
import dgl
import torch


def load_data(data_path):
    data_files = []
    dir_list = sorted(os.listdir(data_path))

    for dirname in dir_list:
        subdir_path = os.path.join(data_path, dirname)
        for filename in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, filename)
            data_files.append(file_path)

    return data_files


class GraphDataset(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.file_list = load_data(root_dir)
        self.size = len(self.file_list)

    def __getitem__(self, index):
        """Get a single data sample by index."""
        file_path = self.file_list[index]
        data = self.loader(file_path)
        return data

    def __len__(self):
        """Return the total number of samples."""
        return self.size


def collate(data):
    graphs = []
    target_students = []
    target_questions = []
    target_responses = []
    student_aliases = []
    question_aliases = []
    question_exists = []

    for sample in data:
        graphs.append(sample[0][0])
        target_students.append(sample[1]['student'])
        target_questions.append(sample[1]['target_question'])
        target_responses.append(sample[1]['target_res'])
        student_aliases.append(sample[1]['student_alias'])
        question_aliases.append(sample[1]['question_alias'])
        question_exists.append(sample[1]['target_question_exist'])

    batch_graph = dgl.batch(graphs)
    target_student = torch.Tensor(target_students).long()
    target_question = torch.Tensor(target_questions).long()
    target_res = torch.Tensor(target_responses).long()
    student_alias = torch.Tensor(student_aliases).long()
    question_alias = torch.Tensor(question_aliases).long()
    question_exist = torch.Tensor(question_exists).long()

    return (batch_graph, target_student, target_question, target_res,
            student_alias, question_alias, question_exist)
