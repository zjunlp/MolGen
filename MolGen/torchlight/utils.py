"""
Utilizations for common usages.
"""
import os
import random
import torch
import numpy as np
from difflib import SequenceMatcher
from unidecode import unidecode
from datetime import datetime
from torch.nn.parallel import DataParallel, DistributedDataParallel


def invert_dict(d):
    return {v: k for k, v in d.items()}

def personal_display_settings():
    """
    Pandas Doc
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.set_option.html
    NumPy Doc
        -
    """
    from pandas import set_option
    set_option('display.max_rows', 500)
    set_option('display.max_columns', 500)
    set_option('display.width', 2000)
    set_option('display.max_colwidth', 1000)
    from numpy import set_printoptions
    set_printoptions(suppress=True)


def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    """
    German and Frence have different vowels than English.
    This utilization removes all the non-unicode characters.
    Example:
        āáǎà  -->  aaaa
        ōóǒò  -->  oooo
        ēéěè  -->  eeee
        īíǐì  -->  iiii
        ūúǔù  -->  uuuu
        ǖǘǚǜ  -->  uuuu

    :param s: unicode string
    :return:  unicode string with regular English characters.
    """
    s = s.strip().lower()
    s = unidecode(s)
    return s


def snapshot(model, epoch, save_path):
    """
    Saving models w/ its params.
        Get rid of the ONNX Protocal.
    F-string feature new in Python 3.6+ is used.
    """
    os.makedirs(save_path, exist_ok=True)
    # timestamp = datetime.now().strftime('%m%d_%H%M')
    save_path = os.path.join(save_path, f'{type(model).__name__}_{epoch}_epoch.pkl')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    return save_path


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'models': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def load_checkpoint(path, map_location):
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint


def show_params(model):
    """
    Show models parameters for logging.
    """
    for name, param in model.named_parameters():
        print('%-16s' % name, param.size())


def longest_substring(str1, str2):
    # initialize SequenceMatcher object with input string
    seqMatch = SequenceMatcher(None, str1, str2)

    # find match of longest sub-string
    # output will be like Match(a=0, b=0, size=5)
    match = seqMatch.find_longest_match(0, len(str1), 0, len(str2))

    # print longest substring
    return str1[match.a: match.a + match.size] if match.size != 0 else ""


def pad(sent, max_len):
    """
    syntax "[0] * int" only works properly for Python 3.5+
    Note that in testing time, the length of a sentence
    might exceed the pre-defined max_len (of training data).
    """
    length = len(sent)
    return (sent + [0] * (max_len - length))[:max_len] if length < max_len else sent[:max_len]


def to_cuda(*args, device=None):
    """
    Move Tensors to CUDA. 
    If no device provided, default to the first card in CUDA_VISIBLE_DEVICES.
    """
    assert all(torch.is_tensor(t) for t in args), \
        'Only support for tensors, please check if any nn.Module exists.'
    if device is None:
        device = torch.device('cuda:0')
    return [None if x is None else x.to(device) for x in args]


def get_code_version(short_sha=True):
    from subprocess import check_output, STDOUT, CalledProcessError
    try:
        sha = check_output('git rev-parse HEAD', stderr=STDOUT,
                           shell=True, encoding='utf-8')
        if short_sha:
            sha = sha[:7]
        return sha
    except CalledProcessError:
        # There was an error - command exited with non-zero code
        pwd = check_output('pwd', stderr=STDOUT, shell=True, encoding='utf-8')
        pwd = os.path.abspath(pwd).strip()
        print(f'Working dir {pwd} is not a git repo.')


def cat_ragged_tensors(left, right):
    assert left.size(0) == right.size(0)
    batch_size = left.size(0)
    max_len = left.size(1) + right.size(1)

    len_left = (left != 0).sum(dim=1)
    len_right = (right != 0).sum(dim=1)

    left_seq = left.unbind()
    right_seq = right.unbind()
    # handle zero padding
    output = torch.zeros((batch_size, max_len), dtype=torch.long, device=left.device)
    for i, row_left, row_right, l1, l2 in zip(range(batch_size),
                                              left_seq, right_seq,
                                              len_left, len_right):
        l1 = l1.item()
        l2 = l2.item()
        j = l1 + l2
        # concatenate rows of ragged tensors
        row_cat = torch.cat((row_left[:l1], row_right[:l2]))
        # copy to empty tensor
        output[i, :j] = row_cat
    return output


def topk_accuracy(inputs, labels, k=1, largest=True):
    assert len(inputs.size()) == 2
    assert len(labels.size()) == 2
    _, indices = inputs.topk(k=k, largest=largest)
    result = indices - labels  # boardcast
    nonzero_count = (result != 0).sum(dim=1, keepdim=True)
    num_correct = (nonzero_count != result.size(1)).sum().item()
    num_example = inputs.size(0)
    return num_correct, num_example


def get_total_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(normalize('ǖǘǚǜ'))
