def data2id():
    import os
    from config import data_dir
    from gcake.data_helper import DataHelper
    datasets = ["FB15K-237", "WN18RR"]
    for dataset in datasets:
        data_helper = DataHelper(dataset)
        with open(os.path.join(data_dir, dataset, "entity2id.txt"), "w", encoding='utf-8') as f:
            f.write(f'{len(data_helper.entity2id)}\n')
            f.write("".join([f"{entity}\t{id}\n" for entity, id in data_helper.entity2id.items()]))
        with open(os.path.join(data_dir, dataset, "relation2id.txt"), "w", encoding='utf-8') as f:
            f.write(f'{len(data_helper.relation2id)}\n')
            f.write("".join([f"{entity}\t{id}\n" for entity, id in data_helper.relation2id.items()]))
        for data_type in ["train", 'valid', 'test']:
            triples, sentences = data_helper.get_data(data_type)
            dst_path = os.path.join(data_dir, dataset, f'{data_type}2id.txt')
            with open(dst_path, "w", encoding="utf-8") as f:
                f.write(f"{len(triples)}\n")
                f.write("".join([f"{h}\t{t}\t{r}\n" for h, r, t in triples]))


def gake():
    dataset = "WN18RR"
    model_name = "GAKE"
    mode = "train"
    from gcake.trainer import GraphTrainer
    GraphTrainer(dataset, model_name).run(mode)


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
    # gake()
    data2id()


if __name__ == '__main__':
    """ 代码执行入口 
        python3 -m ipdb dev.py 
    """

    main()
