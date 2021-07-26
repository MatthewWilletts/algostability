import argparse
import os
import sys

import numpy as np
import torch
import yaml

from runners.cluster_real_data_runner import train, compute_representations, plot_representation, compute_mcc, plot_recons, analyse_lambda
from data.utils import NaNinLoss
import copy


def parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--config', type=str, default='mnist.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--n-sims', type=int, default=0, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--baseline', action='store_true', help='Run the script for the baseline')
    parser.add_argument('--ivae', action='store_true', help='make baseline iVAE')
    parser.add_argument('--representation', action='store_true',
                        help='Run CCA representation validation across multiple seeds')
    parser.add_argument('--mcc', action='store_true', help='compute MCCs -- '
                                                           'only relevant for representation experiments')
    parser.add_argument('--second-seed', type=int, default=0, help='Second random seed for computing MCC -- '
                                                                   'only relevant for representation experiments')
    parser.add_argument('--all', action='store_true',
                        help='Run experiments for many seeds and subset sizes')
    parser.add_argument('--plot', action='store_true',
                        help='Plot selected experiment for the selected dataset')
    parser.add_argument('-z', type=int, default=0)
    args = parser.parse_args()
    return args


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def create_file_with_index(path, suffix='.p'):
    made = False
    counter = 1
    base_path = path
    current_path = base_path + str(suffix)
    while made == False:
        if not os.path.isfile(current_path):
            made = True
            return current_path
        else:
            current_path = base_path + '_' + str(counter) + str(suffix)
            if not os.path.isfile(current_path):
                made = True
                return current_path
            else:
                counter += 1


def make_and_set_dirs(args, config):
    """call after setting args.doc to set and create necessary folders"""
    args.dataset = config.data.dataset.split('_')[0] 
    if 'doc' in vars(args).keys():
        if config.model.final_layer:
            args.doc += str(config.model.feature_size)
        args.doc += config.model.architecture.lower()
    else:
        # args has no attribute doc
        args.doc = config.model.architecture.lower()
    args.doc = os.path.join(args.dataset, args.doc)  # group experiments by dataset
    if 'doc2' in vars(args).keys():
        # add second level doc folder
        args.doc2 = os.path.join(args.doc, args.doc2)
    else:
        # if not defined, set to level 1 doc
        args.doc2 = args.doc
    os.makedirs(args.run, exist_ok=True)
    args.log = os.path.join(args.run, 'logs', args.doc2)
    os.makedirs(args.log, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.doc2)
    os.makedirs(args.checkpoints, exist_ok=True)
    args.output = os.path.join(args.run, 'output', args.doc)
    os.makedirs(args.output, exist_ok=True)

    if 'doc_baseline' in vars(args).keys():
        if config.model.final_layer:
            args.doc_baseline += str(config.model.feature_size)
        args.doc_baseline += config.model.architecture.lower()
        args.doc_baseline = os.path.join(args.dataset, args.doc_baseline)
        args.checkpoints_baseline = os.path.join(args.run, 'checkpoints', args.doc_baseline)
        os.makedirs(args.checkpoints_baseline, exist_ok=True)
        args.output_baseline = os.path.join(args.run, 'output', args.doc_baseline)
        os.makedirs(args.output_baseline, exist_ok=True)
    if 'doc_baseline2' in vars(args).keys():
        if config.model.final_layer:
            args.doc_baseline2 += str(config.model.feature_size)
        args.doc_baseline2 += config.model.architecture.lower()
        args.doc_baseline2 = os.path.join(args.dataset, args.doc_baseline2)
        args.checkpoints_baseline2 = os.path.join(args.run, 'checkpoints', args.doc_baseline2)
        os.makedirs(args.checkpoints_baseline2, exist_ok=True)
        args.output_baseline2 = os.path.join(args.run, 'output', args.doc_baseline2)
        os.makedirs(args.output_baseline2, exist_ok=True)


def main():
    if torch.cuda.is_available():
        dev = torch.cuda.device_count() - 1
        print("Running on gpu:{}".format(dev))
        # torch.cuda.set_device(dev)
    else:
        print("Running on cpu")
    args = parse()
    # load config
    with open(os.path.join('configs', args.config), 'r') as f:
        print('loading config file: {}'.format(os.path.join('configs', args.config)))
        config_raw = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config_raw)
    config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(config)
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.z > 0:
        config.model.final_layer = True
        config.model.feature_size = args.z


    if args.representation:
        config.n_labels = 10
        if not args.mcc and not args.baseline and not args.plot:
            for seed in range(args.seed, args.n_sims + args.seed):
                print('Learning representation for {} - seed: {}'.format(config.data.dataset, seed))
                new_args = argparse.Namespace(**vars(args))
                new_args.seed = seed
                np.random.seed(seed)
                torch.manual_seed(seed)
                new_args.doc = 'representationVADE'
                new_args.doc2 = 'seed{}'.format(seed)
                make_and_set_dirs(new_args, config)
                try:
                    compute_representations(new_args, config)
                except NaNinLoss:
                    print('nans found in run ', seed)

        if args.baseline and not args.mcc and not args.plot:
            if args.ivae:
                for seed in range(args.seed, args.n_sims + args.seed):
                    print('Learning iVAE baseline representation for {} - seed: {}'.format(config.data.dataset, seed))
                    new_args = argparse.Namespace(**vars(args))
                    new_args.seed = seed
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    new_args.doc = 'representationiVAE'
                    new_args.doc2 = 'seed{}'.format(seed)
                    make_and_set_dirs(new_args, config)
                    try:
                        compute_representations(new_args, config, conditional=False)
                    except NaNinLoss:
                        print('nans found in run ', seed)
        if args.mcc and not args.baseline and not args.plot:
            if args.all:
                for seed in range(args.seed, args.n_sims + args.seed - 1):
                    for second_seed in range(seed + 1, args.n_sims + args.seed):
                        print('Computing MCCs for {} - seeds: {} and {}'.format(config.data.dataset, seed,
                                                                                second_seed))
                        new_args = argparse.Namespace(**vars(args))
                        new_args.seed = seed
                        new_args.second_seed = second_seed
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        new_args.doc = 'representationVADE'
                        make_and_set_dirs(new_args, config)
                        for cca_dim in [20]:
                            print('cca dim is ', cca_dim)
                            compute_mcc(new_args, config, cca_dim)
            else:
                assert 'second_seed' in vars(args).keys()
                print('Computing MCCs for {} - seeds: {} and {}'.format(config.data.dataset, args.seed,
                                                                        args.second_seed))
                args.doc = 'representationVADE'
                make_and_set_dirs(args, config)
                compute_mcc(args, config)

        if args.mcc and args.baseline and not args.plot:
            if args.all:
                if args.ivae:
                    for seed in range(args.seed, args.n_sims + args.seed - 1):
                        for second_seed in range(seed + 1, args.n_sims + args.seed):
                            print('Computing baseline MCCs for {} - seeds: {} and {}'.format(config.data.dataset, seed,
                                                                                             second_seed))
                            new_args = argparse.Namespace(**vars(args))
                            new_args.seed = seed
                            new_args.second_seed = second_seed
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            new_args.doc = 'representationiVAE'
                            make_and_set_dirs(new_args, config)
                            for cca_dim in [20]:
                                print('cca dim is ', cca_dim)
                                compute_mcc(new_args, config, cca_dim)
        if args.plot:
            print('Plotting representation experiment for {}'.format(config.data.dataset))
            old_args = copy.deepcopy(args)
            args.doc = 'representationVADE'
            args.doc_baseline = 'representationiVAE'
            make_and_set_dirs(args, config)
            if args.mcc:
                for cca_dim in [20]:
                    print('cca dim is ', cca_dim)
                    plot_representation(args, config, cca_dim=cca_dim)
            config.n_labels = 10 if config.data.dataset.lower().split('_')[0] != 'cifar100' else 100
            if not args.mcc and not args.baseline:
                for seed in range(args.seed, args.n_sims + args.seed):
                    print('Learning representation for {} - seed: {}'.format(config.data.dataset, seed))
                    new_args = argparse.Namespace(**vars(old_args))
                    new_args.seed = seed
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    new_args.doc = 'representationVADE'
                    new_args.doc2 = 'seed{}'.format(seed)
                    make_and_set_dirs(new_args, config)
                    analyse_lambda(new_args, config)
                    plot_recons(new_args, config)
            if args.baseline and not args.mcc:
                if args.ivae:
                    for seed in range(args.seed, args.n_sims + args.seed):
                        print('Learning iVAE baseline representation for {} - seed: {}'.format(config.data.dataset, seed))
                        new_args = argparse.Namespace(**vars(old_args))
                        new_args.seed = seed
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        new_args.doc = 'representationiVAE'
                        new_args.doc2 = 'seed{}'.format(seed)
                        make_and_set_dirs(new_args, config)
                        plot_recons(new_args, config)

if __name__ == '__main__':
    sys.exit(main())
