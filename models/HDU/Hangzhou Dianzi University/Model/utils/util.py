# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL


def out_transform(minibatch_output):
    return


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, args):
    save_dict = {
        "args": vars(args),
        "model_dict": alg.cpu().state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'Mixup': ['class'],
                 'MLDG': ['total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty'],
                 'DIFEX': ['class', 'dist', 'exp', 'align', 'total'],
                 }
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'MAGNET':
        domains = ['3C90', '3C94', '3E6', '3F4', '77', '78', 'N27', 'N30', 'N49', 'N87']
    elif dataset == 'MAGNET_3':
        domains = ['3C90', '3C94', '3E6', '3F4']
    elif dataset == 'MAGNET_7':
        domains = ['77', '78']
    elif dataset == 'MAGNET_N':
        domains = ['N27', 'N30', 'N49', 'N87']
    elif dataset == 'MAGNET_3C90':
        domains = ['3C90']
    elif dataset == 'MAGNET_3C94':
        domains = ['3C94']
    elif dataset == 'MAGNET_3E6':
        domains = ['3E6']
    elif dataset == 'MAGNET_3F4':
        domains = ['3F4']
    elif dataset == 'MAGNET_77':
        domains = ['77']
    elif dataset == 'MAGNET_78':
        domains = ['78']
    elif dataset == 'MAGNET_N27':
        domains = ['N27']
    elif dataset == 'MAGNET_N30':
        domains = ['N30']
    elif dataset == 'MAGNET_N49':
        domains = ['N49']
    elif dataset == 'MAGNET_N87':
        domains = ['N87']
    elif dataset == 'MAGNET_A':
        domains = ['Material A']
    elif dataset == 'MAGNET_B':
        domains = ['Material B']
    elif dataset == 'MAGNET_C':
        domains = ['Material C']
    elif dataset == 'MAGNET_D':
        domains = ['Material D']
    elif dataset == 'MAGNET_E':
        domains = ['Material E']
    elif dataset == 'MAGNET__':
        domains = ['N27', 'N30', 'N49']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'Real_World'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'MAGNET': ['3C90', '3C94', '3E6', '3F4', '77', '78', 'N27', 'N30', 'N49', 'N87'],
        'MAGNET_3': ['3C90', '3C94', '3E6', '3F4'],
        'MAGNET_7': ['77', '78'],
        'MAGNET_N': ['N27', 'N30', 'N49', 'N87'],
        'MAGNET_3C90': ['3C90'],
        'MAGNET_3C94': ['3C94'],
        'MAGNET_3E6': ['3E6'],
        'MAGNET_3F4': ['3F4'],
        'MAGNET_77': ['77'],
        'MAGNET_78': ['78'],
        'MAGNET_N27': ['N27'],
        'MAGNET_N30': ['N30'],
        'MAGNET_N49': ['N49'],
        'MAGNET_N87': ['N87'],
        'MAGNET_A': ['Material A'],
        'MAGNET_B': ['Material B'],
        'MAGNET_C': ['Material C'],
        'MAGNET_D': ['Material D'],
        'MAGNET_E': ['Material E'],
        'MAGNET__': ['N27', 'N30', 'N49']
    }
    if dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    elif dataset == 'MAGNET':
        args.input_shape = (3, 32, 32)
        args.num_classes = 1
    elif dataset == 'MAGNET__':
        args.input_shape = (3, 32, 32)
        args.num_classes = 1
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == 'office-home':
            args.num_classes = 65
        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'PACS':
            args.num_classes = 7
        elif args.dataset == 'VLCS':
            args.num_classes = 5
    return args


def init_norm_dict(npz_path):
    data = np.load(npz_path)

    std_freq_log10 = data['std_freq_log10'][:, np.newaxis]
    mean_freq_log10 = data['mean_freq_log10'][:, np.newaxis]
    std_freq = data['std_freq'][:, np.newaxis]
    mean_freq = data['mean_freq'][:, np.newaxis]
    std_T = data['std_T'][:, np.newaxis]
    mean_T = data['mean_T'][:, np.newaxis]
    std_B = data['std_B'][:, np.newaxis]
    mean_B = data['mean_B'][:, np.newaxis]
    std_H = data['std_H'][:, np.newaxis]
    mean_H = data['mean_H'][:, np.newaxis]
    std_loss = data['std_loss'][:, np.newaxis]
    mean_loss = data['mean_loss'][:, np.newaxis]

    data = {
        'std_freq_log10': std_freq_log10,
        'mean_freq_log10': mean_freq_log10,
        'std_freq': std_freq,
        'mean_freq': mean_freq,
        'std_T': std_T,
        'mean_T': mean_T,
        'std_B': std_B,
        'mean_B': mean_B,
        'std_H': std_H,
        'mean_H': mean_H,
        'std_loss': std_loss,
        'mean_loss': mean_loss
    }

    return data
