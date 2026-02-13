from argparse import ArgumentParser
import warnings

def set_opt():
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()

    parser.add_argument('--dataset', default='phy', help='data name')
    parser.add_argument('--batch_size', type=int, default=2000, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden state size')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
    parser.add_argument('--drop1', type=float, default=0.3, help='drop_out')
    parser.add_argument('--drop2', type=float, default=0.3, help='drop_out')
    parser.add_argument('--layer_num', type=int, default=2, help='GNN layer')
    parser.add_argument('--question_max_length', type=int, default=20, help='the max length of question sequence')
    parser.add_argument('--student_max_length', type=int, default=5, help='the max length of student sequence')
    parser.add_argument('--student_num', type=int, default=795, help='the number of students')
    parser.add_argument('--question_num', type=int, default=720, help='the number of questions')
    parser.add_argument('--skill_num', type=int, default=35, help='the number of skills')
    parser.add_argument('--patience', type=int, default=5, help='early stop patience')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cv_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1010)


    opt = parser.parse_args()

    return opt
