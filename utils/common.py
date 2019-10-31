# Shared functions between each subtask manager (if any)
import argparse
import logging
import time
import os
from setproctitle import setproctitle

import numpy
import torch


def add_base_args(parser):
    ''' general arguments shared between different model training '''
    parser.add_argument('--log-subdir', type=str, default='', metavar='path',
                        help='Set log subdirectory (all log will be store under ./log)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='How many batches to wait before logging training status')
    parser.add_argument('--test-interval', type=int, default=100, metavar='N',
                        help='How many batches to test during training')
    parser.add_argument('--process-name', default='Hello World!',
                        type=str, help='Used to hide process name')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    # For continuous training
    # e.g. Use the multitask KGE during TE training as transfer learning base model
    # e.g. Process been shutdown unexpectedly
    pretrain_group = parser.add_mutually_exclusive_group()
    pretrain_group.add_argument('--pretrain-ckpt', type=str, default='',
                                help='If specific, copy the ckpt as pretrain model')
    pretrain_group.add_argument('--pretrain-latest', action='store_true', default=False,
                                help='Find the latest ckpt of the same model and dataset')


def set_random_seed(configs):
    ''' set random seed '''
    numpy.random.seed(configs.seed)
    torch.manual_seed(configs.seed)


def set_additional_args(args, configs):
    ''' set additional arguments to args '''
    ctime = time.localtime()
    args.date = f'{ctime.tm_mon}-{ctime.tm_mday}_{ctime.tm_hour}-{ctime.tm_min}'
    # set log directory
    if not args.log_subdir:
        args.log_subdir = args.date
    args.log_dir = os.path.join('log', configs.subtask, args.log_subdir)
    os.makedirs(args.log_dir, exist_ok=True)
    # set ckpt directory
    args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
    os.makedirs(args.ckpt_dir)
    # set tensorboard log
    args.tb_dir = os.path.join(args.log_dir, args.ckpt_name+'_tensorboard')
    os.makedirs(args.tb_dir)


def setup_log(args, logfile_level=logging.DEBUG, stdout_level=logging.INFO):
    ''' setup log '''
    logfile = os.path.join(args.log_dir, f'{args.log_process_name}.log')
    # configure log
    logging.basicConfig(level=logfile_level,
                        format='%(asctime)s %(name)-13s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logfile,
                        filemode='w')
    # handler for stdout
    console = logging.StreamHandler()
    console.setLevel(stdout_level)
    formatter = logging.Formatter('%(name)-13s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def set_process_name(args):
    ''' set process name '''
    if args.process_name:
        setproctitle(args.log_process_name)
    else:
        setproctitle(proctitle)


def check_gpu(args):
    ''' check if gpu is available and log info '''
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    logging.info(f'Use device: {device}')
    if use_cuda:
        logging.info('\tDevices: {}, Current Device: #{}-{}'.format(
            torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name()))
        logging.info('current memory allocated: {}MB'.format(
            torch.cuda.memory_allocated() / 1024 ** 2))
        logging.info('max memory allocated: {}MB'.format(
            torch.cuda.max_memory_allocated() / 1024 ** 2))
        logging.info('cached memory: {}MB'.format(
            torch.cuda.memory_cached() / 1024 ** 2))
    return use_cuda
