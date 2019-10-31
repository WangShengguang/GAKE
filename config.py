class Config(object):
    """ this is going to contain all the config of model """
    subtask = 'general'
    seed = 87
    batch_size = 64
    test_batch_size = 128
    lr = 0.0001


class DevConfig(object):
    """ only used for development purpose """
    subtask = 'dev'
    seed = 87
    batch_size = 64
    test_batch_size = 128
    lr = 0.0001
