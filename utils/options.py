import argparse
import os
import torch
import models
import time

class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--log_dir', type=str, default='./logs', help='logs are saved here')
        parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--save_model', type=bool, default=False, help='whether to save the model')
        parser.add_argument('--task', type=str, default='meta', help='meta')
        parser.add_argument('--gpu_ids', type=list, default=[0],  help='indicates the gpus will be used.')
        # model parameters
        parser.add_argument('--model', type=str, default='ctmwa', help='chooses which model to use. [self_learning_vit_bert | self_learning | self_learning_v2]')
        parser.add_argument('--pretrained', type=str, default='bert-base-uncased', help='Bert pretrained')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012, help='scaling factor for normal, xavier and orthogonal.')
        # dataset parameters
        parser.add_argument('--dataset', type=str, default='tum_emo', help='which dataset to use. [mvsa_single | mvsa_mul | tum_emo]')
        # training parameter
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        self.isTrain = True
        self.initialized = True

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message +='----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        if opt.verbose:
            print(message)

        # log dir
        log_dir = os.path.join(opt.log_dir, f'{opt.dataset}_{opt.model}')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_dir = os.path.join(log_dir, str(opt.seed))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # result dir
        result_dir = os.path.join(opt.result_dir, f'{opt.dataset}')
        
        result_dir = os.path.join(result_dir, f'{opt.model}_{opt.seed}_{time.strftime("%Y-%m-%d-%H.%M.%S",time.localtime(time.time()))}')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        opt.result_dir = result_dir

        # save opt as txt file
        file_name = os.path.join(result_dir, '{}_opt.txt'.format(opt.task))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, seed):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.seed = seed
        # process opt.suffix
        opt.name = f'{opt.dataset}_{opt.model}'
        print("Expr Name:", opt.name)
        
        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            
        self.opt = opt
        return self.opt
