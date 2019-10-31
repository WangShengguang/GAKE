import argparse


def gake():
    dataset = "WN18RR"
    model_name = "GAKE"
    mode = "train"
    from gcake.models.gake.gake import Trainer
    Trainer(dataset).run(mode)


def main():
    # ''' Parse command line arguments and execute the code
    #     --stream_log, --relative_path, --log_level
    #     --allow_gpus, --cpu_only
    # '''
    # parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    # group.add_argument('--model', type=str, choices=["GAKE"], help="model name")
    # parser.add_argument('--mode', type=str, choices=["train", "test", "valid"],
    #                     required=True, help="训练or测试")
    # parser.add_argument('--dataset', type=str, choices=["FB15K-237", "WN18RR"])
    # # parse args
    # args = parser.parse_args()
    # # mode = "train" if args.train else "test"
    # mode = args.mode
    gake()


if __name__ == '__main__':
    """ 代码执行入口 
        python3 -m ipdb dev.py 
    """

    main()
