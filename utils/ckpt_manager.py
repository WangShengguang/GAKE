# Checkopint index manager
import yaml
import os
import logging

logger = logging.getLogger('ckpt_manager')

CKPT_INDEX = 'log/{}/ckpt_index.yaml'

EMPTY_INDICES = {
    'latest': None,
    'history': []
}

EMPTY_INDEX = {
    'date': None,
    'ckpt_dir': None,
    # to be determined
    'train_epoch': None,
    'max_acc': None,
    'min_loss': None
}

BEST_METRICS = {
    'train_epoch': max,
    'max_acc': max,
    'min_loss': min
}


class CheckpointManager:
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.ckpt_file = CKPT_INDEX.format(configs.subtask)
        if not os.path.exists(self.ckpt_file):
            os.makedirs(os.path.dirname(self.ckpt_file), exist_ok=True)
            self.ckpt = {}
            self.write_ckpt()
        else:
            self.load_ckpt()
        self.record = None

    def load_ckpt(self):
        ''' load checkpoint yaml into memory '''
        with open(self.ckpt_file, 'r') as stream:
            self.ckpt = yaml.safe_load(stream)

    def write_ckpt(self):
        ''' write checkpoint back into yaml '''
        with open(self.ckpt_file, 'w') as stream:
            yaml.safe_dump(self.ckpt, stream)

    def _check_ckpt_exist(self):
        ''' check if ckpt exist '''
        if not self.ckpt.get(self.args.ckpt_name):
            logger.warning(f'No {self.args.ckpt_name} checkpoint found')
            self.ckpt[self.args.ckpt_name] = EMPTY_INDICES.copy()
            return False
        else:
            return True

    def get_latest_ckpt(self):
        ''' get latest checkpoint for a specific model trained with a specific dataset '''
        self._check_ckpt_exist()
        latest_ckpt = self.ckpt[self.args.ckpt_name].get('latest')
        if not latest_ckpt:
            logger.warning(
                f'No latest checkpoint for {self.args.ckpt_name} found')
            return EMPTY_INDEX.copy()
        else:
            return latest_ckpt

    def get_best_ckpt(self, metrics):
        ''' get the best checkpoint based on metrics '''
        self._check_ckpt_exist()
        history_ckpt = self.ckpt[self.args.ckpt_name].get('history')
        if not history_ckpt:
            logger.warning(
                f'No history checkpoint for {self.args.ckpt_name} found')
            return EMPTY_INDEX.copy()
        else:
            return BEST_METRICS[metrics](history_ckpt, key=lambda x: x[metrics])

    def add_new_ckpt_index(self):
        ''' create and add a new checkpoint index into indices '''
        self._check_ckpt_exist()
        # manage the "latest" record
        self.record = EMPTY_INDEX.copy()
        self.record['date'] = self.args.date
        self.record['ckpt_dir'] = self.args.ckpt_dir

        self.ckpt[self.args.ckpt_name]['latest'] = self.record
        self.history_idx = len(self.ckpt[self.args.ckpt_name]['history'])
        self.ckpt[self.args.ckpt_name]['history'].append(self.record)
        self.write_ckpt()

    def load_old_ckpt_record(self):
        ''' load old checkpoint record from indices '''
        for i, candidate_record in enumerate(self.ckpt[self.args.ckpt_name]['history']):
            if candidate_record['ckpt_dir'] == self.args.ckpt_dir:
                self.record = candidate_record
                self.history_idx = i
                break
        if self.record is None:
            logger.warning(f'No record of {self.args.ckpt_dir} found')

    def update_ckpt_index(self, **kargs):
        for key, value in kargs.items():
            self.record[key] = value
        self.ckpt[self.args.ckpt_name]['history'][self.history_idx] = self.record
        self.write_ckpt()


if __name__ == "__main__":
    # test functionality
    class Args:
        def __init__(self, ckpt_name, date, ckpt_dir):
            self.ckpt_name = ckpt_name
            self.date = date
            self.ckpt_dir = ckpt_dir

    class Configs:
        subtask = 'test'

    args1 = Args('BERT_CRF_BERT_TransKB_lawdata', '10-21_09-51',
                 'log/10-21_09-51/BERT_CRF_BERT_TransKB_lawdata/ckpt')

    # same model same dataset different date, checkpoint
    args2 = Args('BERT_CRF_BERT_TransKB_lawdata', '10-21_10-23',
                 'log/10-21_10-23/BERT_CRF_BERT_TransKB_lawdata/ckpt')

    # same model, checkpoint different date
    args3 = Args('BERT_CRF_BERT_TransKB_lawdata', '10-21_10-36',
                 'log/10-21_09-51/BERT_CRF_BERT_TransKB_lawdata/ckpt')

    configs = Configs()

    manager1 = CheckpointManager(args1, configs)
    print(manager1.get_best_ckpt('max_acc'))  # this should get a warning
    manager1.add_new_ckpt_index()
    manager1.update_ckpt_index(**{
        'train_epoch': 10,
        'max_acc': 0.87,
        'min_loss': 0.03
    })
    manager2 = CheckpointManager(args2, configs)
    manager2.add_new_ckpt_index()
    manager2.update_ckpt_index(**{
        'train_epoch': 3,
        'max_acc': 0.64,
        'min_loss': 0.42
    })
    print(manager2.get_latest_ckpt())
    manager3 = CheckpointManager(args1, configs)
    manager3.load_old_ckpt_record()
    print(manager3.record)
    print(manager3.history_idx)
    print(manager1.get_best_ckpt('min_loss'))
