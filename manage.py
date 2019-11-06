# Manager for KGE
import argparse
import logging

from config import Config
from utils import common
import sys


def set_names(args):
    ''' set names for log, process and checkpoint '''
    # get log filename and process name
    args.log_process_name = "_".join([args.dataset, args.model, args.mode])
    args.process_name = f"{sys.argv[0]} {args.dataset[0]} {args.model[0]}{args.model[-1]}"
    # get checkpoint name
    args.ckpt_name = '{}_{}'.format(
        args.dataset, args.model
    )


def print_settings(args, configs: Config):
    ''' logging some information into log '''
    logging.info('Configurations:')
    logging.info(f'\tDataset\t\t: {args.dataset}')
    logging.info(f'\tMode\t\t: {args.mode}')
    logging.info(f'\tUsing Model:')
    logging.info(f'\t Main Model: {args.model}')
    logging.info(f'\t\tSentence Encoder: {configs.sentence_encoder}')
    logging.info(f'\t\tGraph Encoder: {configs.graph_encoder}')
    logging.info(f'\tParameters:')
    # if use Adam then remove
    logging.info(f'\t Learning Rate\t: {configs.lr}')
    logging.info(f'\t Train Batch\t: {configs.batch_size}')
    logging.info(f'\t Test Batch\t: {configs.test_batch_size}')
    logging.info(f'Path:')
    logging.info(f'\tlog: {args.log_dir}')
    logging.info(f'\tckpt: {args.ckpt_dir}')


def parse_argument():
    parser = argparse.ArgumentParser(
        description='GCAKE Manager')
    parser.add_argument('--dataset', type=str, default='FB15K-237',
                        choices=['FB15K-237', 'WN18RR'],
                        help='Dataset')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode to use')
    parser.add_argument('--model', type=str, default='GCAKE',
                        choices=['GAKE', 'GCAKE'],
                        help='Model for KGE')
    parser.add_argument('--use_graph', action="store_true", help="是否使用Graph Embedding")

    common.add_base_args(parser)
    return parser.parse_args()


def run(args, configs):
    from gcake.trainer import Trainer, GraphTrainer
    from gcake.constructor import Constructor
    model = Constructor(args, configs).get_model()
    if args.model == "GAKE":
        trainer = GraphTrainer(model, args, configs)
    else:
        trainer = Trainer(model, args, configs)
    trainer.run(args.mode)


def main():
    # get args and configs
    args = parse_argument()
    configs = Config()

    set_names(args)
    common.setup_log(args.log_process_name)
    common.set_random_seed(configs)
    common.set_additional_args(args, configs)
    common.set_process_name(args)
    use_cuda = common.check_gpu(args)
    print_settings(args, configs)

    run(args, configs)

    # If use cuda and have multiple GPU, then warp model with nn.DataParallel
    # the following code is going to be put in Trainer?!
    #
    # if use_cuda:
    #     if torch.cuda.device_count() > 1:
    #         logging.info('Model running on multiple GPUs ({})'.format(
    #             torch.cuda.device_count()))
    #         ner_model = torch.nn.DataParallel(ner_model)
    #         re_model = torch.nn.DataParallel(re_model)
    #         kge_model = torch.nn.DataParallel(kge_model)
    #     else:
    #         logging.info('Model running on single GPU')
    #
    # # get tensorboard writer
    # tbwriter = SummaryWriter(logdir=args.tb_dir)


if __name__ == '__main__':
    main()
