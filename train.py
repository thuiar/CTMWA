import os
from torch.utils.data import DataLoader
from utils.dataset import BaseDataset
from models import create_model
from utils import setup_seed, Options, get_logger
import torch.nn as nn

def train_task(model, tr_loader, qy_loader, vl_loader, ts_loader,opt, logger, index):
    for i in range(opt.niter + opt.niter_decay):
        model.train()
        model.inner_train(tr_loader, qy_loader, logger, i)
        model.eval()
        model.inner_test(vl_loader, logger,'Val')
        model.inner_test(ts_loader, logger, 'Test')
        # model.save_networks(index)

def train_meta(opt):
    opt.criterion_clsloss = nn.CrossEntropyLoss()
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.seed))
    suffix = '_'.join([f'{opt.dataset}_{opt.model}', opt.task])
    logger = get_logger(logger_path, suffix)            # get logger
    setup_seed(opt.seed)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    opt.device = model.device
    # setup the base model.
    index = 1
    train_data = BaseDataset(dataset = opt.dataset, data_type='train', index=index)
    tr_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    qy_data = BaseDataset(dataset = opt.dataset, data_type='query', index=index)
    qy_loader = DataLoader(qy_data, batch_size=opt.batch_size, shuffle=False)
    vl_data = BaseDataset(dataset = opt.dataset, data_type='valid', index=index)
    vl_loader = DataLoader(vl_data, batch_size=opt.batch_size, shuffle=False)
    test_data = BaseDataset( dataset = opt.dataset,data_type='test', index=index)
    ts_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    train_task(model, tr_loader, qy_loader, vl_loader, ts_loader,opt, logger, index)

if __name__ == "__main__":
    seed = 1111
    opt = Options().parse(seed=seed)
    train_meta(opt)
