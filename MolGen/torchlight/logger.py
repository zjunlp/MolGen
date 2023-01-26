import os
import re
import sys
import time
import json
import torch
import pickle
import random
import getpass
import logging
import argparse
import subprocess
import numpy as np
from datetime import timedelta, date
from .utils import get_code_version


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def initialize_exp(params):
    """
    Initialize the experiment:
    - dump parameters
    - create a logger
    """
    # dump parameters
    exp_folder = get_dump_path(params)
    json.dump(vars(params), open(os.path.join(exp_folder, 'params.pkl'), 'w'), indent=4)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(exp_folder, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    # logger.info("\n".join("%s: %s" % (k, str(v))
    #                       for k, v in sorted(dict(vars(params)).items())))
    # text = f'# Git Version: {get_code_version()} #'
    # logger.info("\n".join(['=' * 24, text, '=' * 24]))
    logger.info("The experiment will be stored in %s\n" % exp_folder)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0
    assert not params.dump_path in ('', None), \
            'Please choose your favorite destination for dump.'
    dump_path = params.dump_path

    # create the sweep path if it does not exist
    when = date.today().strftime('%m%d-')
    sweep_path = os.path.join(dump_path, when + params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an random ID for the job if it is not given in the parameters.
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break
        params.exp_id = exp_id

    # create the dump folder / update parameters
    exp_folder = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(exp_folder):
        subprocess.Popen("mkdir -p %s" % exp_folder, shell=True).wait()
    return exp_folder


if __name__ == '__main__':
    pass
