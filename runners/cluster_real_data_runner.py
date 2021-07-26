import os
import pickle

import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.cross_decomposition import CCA
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.utils as utils

from data.utils import single_one_hot_encode, single_one_hot_encode_rev, get_good_dims, NaNinLoss
from data.utils import Fast_CIFAR10, Fast_MNIST, Fast_Fashion_MNIST, Fast_SVHN
from metrics.mcc import mean_corr_coef, mean_corr_coef_out_of_sample
from metrics.acc import cluster_acc_and_conf_mat
from models.nde import VaDEConvMLP, VaDEFullMLP, iVAEConvMLP, iVAEFullMLP, VaDEResNetMLP, iVAEResNetMLP
from models.vqvae import VQVAE, GSSOFT

from .plotting import plot_recons_and_originals, plot_recons_originals_samples, sift_most_confident_images
import boilr
if int(boilr.__version__[2]) == 5 or int(boilr.__version__[2]) == 6 and int(boilr.__version__[4]) < 4:
    from boilr.nn_init import data_dependent_init
else:
    from boilr.nn.init import data_dependent_init

sns.set(font_scale=1.5, rc={'text.usetex' : True, "lines.linewidth": 0.7})


def get_dgm(args, config, good_dims=None):
    if config.model.architecture.lower() == 'vqvae':
        return VQVAE(config)
    if config.model.architecture.lower() == 'rvqvae':
        return GSSOFT(config)
    if args.ivae == False:
        if config.model.architecture.lower() == 'convmlp':
            print('making VaDEConvMLP')
            print('ncomp: ', config.model.num_components)
            return VaDEConvMLP(config)
        elif config.model.architecture.lower() == 'mlp':
            print('making VaDEFullMLP')
            print('ncomp: ', config.model.num_components)
            return VaDEFullMLP(config, good_dims=good_dims)
        elif config.model.architecture.lower() == 'resnet':
            return VaDEResNetMLP(config, good_dims=good_dims)
        elif config.model.architecture.lower() == 'flow':
            return GMMflow(config)
    else:
        if config.model.architecture.lower() == 'convmlp':
            print('making iVAEConvMLP')
            return iVAEConvMLP(config)
        elif config.model.architecture.lower() == 'mlp':
            print('making iVAEFullMLP')
            return iVAEFullMLP(config, good_dims=good_dims)
        elif config.model.architecture.lower() == 'resnet':
            return iVAEResNetMLP(config, good_dims=good_dims)
        elif config.model.architecture.lower() == 'flow':
            return iGMMflow(config)


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def get_dataset(args, config, test=False, one_hot=True, shuffle=True, on_gpu=True):
    total_labels = 10
    reduce_labels = total_labels != config.n_labels

    if config.data.random_flip is False:
        transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        if not test:
            transform = transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.data.image_size),
                transforms.ToTensor()
            ])
    if test:
        on_gpu = False
    if config.data.dataset.lower().split('_')[0] == 'mnist':
        if on_gpu:
            dataset = Fast_MNIST(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform, device=config.device)
        else:
            dataset = MNIST(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform)
        if config.model.architecture == 'mlp' or config.model.architecture == 'resnet':
            # find good dims
            good_dims = get_good_dims(dataset.data.data.cpu().numpy().reshape(-1,784))
        else:
            good_dims = None

    elif config.data.dataset.lower().split('_')[0] == 'cifar10':
        if on_gpu:
            dataset = Fast_CIFAR10(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform, device=config.device)
        else:
            dataset = CIFAR10(os.path.join(args.run, 'datasets'), train=not test, download=True, transform=transform)
            # dataset.data = np.transpose(dataset.data, (0, 3, 1, 2))
        good_dims = None
    elif config.data.dataset.lower().split('_')[0] == 'svhn':
        if test:
            split = 'test'
        else:
            split = 'train'
        if on_gpu:
            dataset = Fast_SVHN(os.path.join(args.run, 'datasets'), split=split, download=True, transform=transform, device=config.device)
        else:
            dataset = SVHN(os.path.join(args.run, 'datasets'), split=split, download=True, transform=transform)
        good_dims = None
    else:
        raise ValueError('Unknown config dataset {}'.format(config.data.dataset))

    if hasattr(dataset, 'targets'):
        if type(dataset.targets) is list:
            # CIFAR10 storea targets as list, unlike MNIST which uses torch.Tensor
            dataset.targets = np.array(dataset.targets)

    cond_size = config.n_labels

    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=0)

    return dataloader, dataset, cond_size, good_dims


def train(args, config, conditional=True, return_elbo=False):
    save_weights = 'baseline' not in config.data.dataset.lower()  # we don't need the
    if conditional == False:
        config.model.num_components = 1

    print('save_weights', save_weights)
    # load dataset
    dataloader, dataset, cond_size, good_dims = get_dataset(args, config, one_hot=True)
    # define the deep generative model
    model = get_dgm(args, config, good_dims)

    optimizer = get_optimizer(config, model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)

    if 'vqvae' not in config.model.architecture.lower():
        # Get batch
        t = [dataloader.dataset[i] for i in range(config.training.batch_size)]
        t = torch.stack(tuple(t[i][0] for i in range(len(t))))
        # Use batch for data dependent init
        with torch.no_grad():
            data_dependent_init(model, {'x': t})

    # train
    step = 0
    max_steps = len(dataloader) * config.training.n_epochs
    loss_track_epochs = []
    model.train()
    for epoch in range(config.training.n_epochs):
        elbo_train = 0
        KL_z_train = 0
        lxz_train = 0
        for i, (X, y) in enumerate(dataloader):
            step += 1
            X = X.to(config.device)
            if config.data.logit_transform:
                X = logit_transform(X)
            if config.data.dataset == 'MNIST':
                X = torch.bernoulli(X)
            # compute loss
            if args.ivae:
                elbo, elbo_raw, z_est = model.elbo(X, y)
            else:
                elbo, elbo_raw, z_est = model.elbo(X)
            # optimize
            loss = elbo.mul(-1)
            if torch.isnan(loss):
                # write current batch and last batch to file
                pickle.dump(previous_batch,
                    open(os.path.join(args.checkpoints,
                                  'previous_batch_{}.p'.format(args.seed)), 'wb'))
                pickle.dump((X.data.cpu().numpy(), y.data.cpu().numpy()),
                    open(os.path.join(args.checkpoints,
                                  'last_batch_{}.p'.format(args.seed)), 'wb'))
                print('ELBO raw at NaN: ', elbo_raw)
                raise NaNinLoss("Loss is NaN")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('i ', i, ' epoch ', epoch, ' ELBO ', elbo)

            if step >= max_steps and save_weights:
                # save final checkpoints for distribution!
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    scheduler
                ]
                print('saving weights under: {}'.format(args.checkpoints))
                # torch.save(states, os.path.join(args.checkpoints, 'checkpoint_{}.pth'.format(step)))
                torch.save(states, os.path.join(args.checkpoints, 'checkpoint.pth'))

            if step % config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    scheduler,
                ]
                print('checkpoint at step: {}'.format(step))
                print('saving weights under: {}'.format(args.checkpoints))
                # torch.save(states, os.path.join(args.checkpoints, 'checkpoint_{}.pth'.format(step)))
                torch.save(states, os.path.join(args.checkpoints, 'checkpoint.pth'))
            elbo_train += -elbo.item()
            lxz_train += elbo_raw[0].item()
            KL_z_train += elbo_raw[1].item()
            previous_batch = (X.data.cpu().numpy(), y.data.cpu().numpy())
        elbo_train /= len(dataloader)
        lxz_train /= len(dataloader)
        KL_z_train /= len(dataloader)
        print('epoch ', epoch, ' ELBO ', elbo_train, 'LPX ', lxz_train, 'KL z ', KL_z_train)
        loss_track_epochs.append(elbo_train)
        scheduler.step(elbo_train)
        if 'vqvae' not in config.model.architecture.lower():
            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    samples = model.sample(10, per_comp=True)
                    # samples = samples.permute(1, 0, 2)
                    samples = samples.reshape(-1, model.n_channels, model.image_size, model.image_size)
                    if model.num_components > 1:
                        nrow = model.num_components
                    else:
                        nrow = 10
                    grid = utils.make_grid(samples + 0.5, nrow=nrow)
                    utils.save_image(grid, args.checkpoints + '/samples_'+str(epoch)+'.png')

    # save loss track during epoch for transfer baseline
    print('saving loss track under: {}'.format(args.checkpoints))
    print('final ELBO:', elbo_train)
    pickle.dump(loss_track_epochs,
                open(os.path.join(args.checkpoints,
                                  'all_epochs_SEED{}.p'.format(args.seed)), 'wb'))

    # save final checkpoints for distrubution!
    if save_weights:
        states = [
            model.state_dict(),
            optimizer.state_dict(),
            scheduler
        ]
        print('saving weights under: {}'.format(args.checkpoints))
        # torch.save(states, os.path.join(args.checkpoints, 'checkpoint_{}.pth'.format(step)))
        torch.save(states, os.path.join(args.checkpoints, 'checkpoint.pth'))
    if return_elbo:
        with torch.no_grad():
            elbo_train = 0
            for i, (X, y) in enumerate(dataloader):
                step += 1
                model.train()
                X = X.to(config.device)
                if config.data.logit_transform:
                    X = logit_transform(X)
                if config.data.dataset == 'MNIST':
                    X = torch.bernoulli(X)
                # compute loss
                if args.ivae:
                    elbo, elbo_raw, z_est = model.elbo(X, y)
                else:
                    elbo, elbo_raw, z_est = model.elbo(X)
                # optimize

                elbo_train += -elbo.item()
            elbo_train /= len(dataloader)
        return elbo_train


def compute_representations(args, config, conditional=True):
    """
    we train an icebeem model or an unconditional EBM across multiple random seeds and
    compare the reproducibility of representations via CCA

    first we train the entire network, then we save the activations !
    """
    # train the energy model on full train dataset and save feature maps
    save_weights = 'baseline' not in config.data.dataset.lower()  # we don't need the
    if conditional == False:
        config.model.num_components = 1

    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    load_model = os.path.isfile(ckpt_path)

    if config.model.retrain:
        load_model = False
    # if checkpoint doesn't exist, train model
    if load_model == False:
        print('training model')
        ELBO_train = train(args, config, conditional=conditional, return_elbo=True)
        BPD_train = ELBO_train/(config.data.image_size ** 2 * config.data.channels * np.log(2))
    # load test data
    dataloader, dataset, cond_size, test_good_dims = get_dataset(args, config, test=True, one_hot=True, shuffle=False, on_gpu=config.data.on_gpu)
    train_dataloader, train_dataset, cond_size, train_good_dims = get_dataset(args, config, one_hot=True, shuffle=False, on_gpu=False)
    # load feature mapts
    print('loading model weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)

    f = get_dgm(args, config, train_good_dims).to(config.device)

    f.load_state_dict(states[0])

    if 'vqvae' not in config.model.architecture.lower():
        samples = f.sample(10, per_comp=True)
        # samples = samples.permute(1, 0, 2)
        samples = samples.reshape(-1, f.n_channels, f.image_size, f.image_size)
        grid = utils.make_grid(samples + 0.5, nrow=f.num_components)
        utils.save_image(grid, args.checkpoints + '/samples_end.png')


    if load_model:
        with torch.no_grad():
            ELBO_train = 0
            for i, (X, y) in enumerate(train_dataloader):
                f.eval()
                X = X.to(config.device)
                if config.data.logit_transform:
                    X = logit_transform(X)
                if config.data.dataset == 'MNIST':
                    X = torch.bernoulli(X)
                # compute loss
                if args.ivae:
                    elbo, elbo_raw, z_est = f.elbo(X, y)
                else:
                    elbo, elbo_raw, z_est = f.elbo(X)
                ELBO_train += -elbo.item()
            ELBO_train /= len(train_dataloader)
            BPD_train = ELBO_train/(f.input_size * np.log(2))

    # compute and save test features
    if 'vqvae' not in config.model.architecture.lower():
        with torch.no_grad():
            representations = np.zeros((len(dataset), f.latent_dim))
            labels = np.zeros((len(dataset),))
            y_pred = np.zeros((len(dataset),))
            y_logits = np.zeros((len(dataset), f.num_components))
            counter = 0
            ELBO_test = []
            for i, (X, y) in enumerate(dataloader):
                X = X.to(config.device)
                if config.data.logit_transform:
                    X = logit_transform(X)
                if config.data.dataset == 'MNIST':
                    X = torch.bernoulli(X)
                rep_i = f.encode(X)[0].view(-1, f.latent_dim).data.cpu().numpy()
                ELBO_test.append(f.elbo(X, y)[0])
                representations[counter:(counter + rep_i.shape[0])] = rep_i
                labels[counter:(counter + rep_i.shape[0])] = y.data.cpu().numpy()
                if 'representationVADE' in args.doc or 'representationiVAE' in args.doc:
                    # record cluster assignment
                    q_y_x = f.compute_q_y_x(X).data.cpu().numpy()
                    y_logits[counter:(counter + q_y_x.shape[0])] = q_y_x
                    y_pred[counter:(counter + q_y_x.shape[0])] = np.argmax(q_y_x, axis=1)

                counter += rep_i.shape[0]
            representations = representations[:counter]
            labels = labels[:counter]
            num_batches = len(ELBO_test)
            ELBO_test = sum(ELBO_test)/float(num_batches)
            BPD_test = ELBO_test/(f.input_size * np.log(2)) - 7

            if config.data.dataset == 'MNIST':
                sift_most_confident_images(y_logits, dataset.data.cpu().numpy(), args.checkpoints, good_dims=None)
            else:
                sift_most_confident_images(y_logits, dataset.data, args.checkpoints, good_dims=None)

            print('saving test representations under: {}'.format(args.checkpoints))

            if 'representationVADE' in args.doc:
                cluster = cluster_acc_and_conf_mat(labels.astype(int), y_pred.astype(int))
                print('Cluster acc: ', cluster[0])
                pickle.dump({'rep': representations, 'lab': labels, 'y_pred': y_pred,
                            'acc': cluster[0], 'conf_mat': cluster[1], 'ELBO_test': ELBO_test, 'BPD_test': BPD_test,
                             'ELBO_train': ELBO_train, 'BPD_train': BPD_train},
                            open(os.path.join(args.checkpoints, 'test_representations.p'), 'wb'))
            else:
                pickle.dump({'rep': representations, 'lab': labels, 'ELBO_test': ELBO_test, 'BPD_test': BPD_test,
                            'ELBO_train': ELBO_train, 'BPD_train': BPD_train},
                            open(os.path.join(args.checkpoints, 'test_representations.p'), 'wb'))
    elif 'vqvae' in config.model.architecture.lower():
        with torch.no_grad():
            representations = np.zeros((len(dataset), f.embedding_dim, 8, 8))
            counter = 0
            ELBO_test = []
            for i, (X, y) in enumerate(dataloader):
                X = X.to(config.device)
                if config.data.logit_transform:
                    X = logit_transform(X)
                if config.data.dataset == 'MNIST':
                    X = torch.bernoulli(X)
                rep_i = f.encode(X).view(-1, f.embedding_dim, 8, 8).data.cpu().numpy()
                ELBO_test.append(f.elbo(X, y)[0])
                representations[counter:(counter + rep_i.shape[0])] = rep_i
                counter += rep_i.shape[0]
            representations = representations[:counter]
            num_batches = len(ELBO_test)
            ELBO_test = sum(ELBO_test)/float(num_batches)
            BPD_test = ELBO_test/(f.input_size * np.log(2)) - 7

            print('saving test representations under: {}'.format(args.checkpoints))

            pickle.dump({'rep': representations, 'ELBO_test': ELBO_test, 'BPD_test': BPD_test,
                        'ELBO_train': ELBO_train, 'BPD_train': BPD_train},
                        open(os.path.join(args.checkpoints, 'test_representations.p'), 'wb'))


def analyse_lambda(args, config, conditional=True):
    """
    we train an icebeem model or an unconditional EBM across multiple random seeds and
    compare the reproducibility of representations via CCA

    first we train the entire network, then we save the activations !
    """
    # train the energy model on full train dataset and save feature maps
    # load test data
    dataloader, dataset, cond_size, good_dims = get_dataset(args, config, test=True, one_hot=True, shuffle=False)
    # load feature mapts
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading VAE weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)

    f = get_dgm(args, config, good_dims).to(config.device)

    f.load_state_dict(states[0])

    lamba_matrix = f.lambda_values().detach().cpu().numpy()
    raise Exception


def plot_recons(args, config, conditional=True):
    """
    we train an icebeem model or an unconditional EBM across multiple random seeds and
    compare the reproducibility of representations via CCA

    first we train the entire network, then we save the activations !
    """
    # train the energy model on full train dataset and save feature maps
    # load test data
    dataloader, dataset, cond_size, good_dims = get_dataset(args, config, test=True, one_hot=True, shuffle=False)
    # load feature mapts
    ckpt_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    print('loading VAE weights from: {}'.format(ckpt_path))
    states = torch.load(ckpt_path, map_location=config.device)

    f = get_dgm(args, config, good_dims).to(config.device)

    f.load_state_dict(states[0])

    # compute and save test features
    # first batch
    t = [dataloader.dataset[i] for i in range(16)]
    t = torch.stack(tuple(t[i][0] for i in range(len(t))))
    t = t.to(config.device)
    if 'vqvae' not in config.model.architecture.lower():
        f_out, mu, logv = f(t)
    else:
        f_out, kl = f(t)
    print('saving test recons under: {}'.format(args.checkpoints))

    plot_recons_and_originals(f_out.detach(), t.detach(), args.checkpoints, 'test')
    if 'vqvae' not in config.model.architecture.lower():
        if conditional:
            samples = f.sample(10, per_comp=True)
            # samples = samples.permute(1, 0, 2)
            samples = samples.reshape(-1, f.n_channels, f.image_size, f.image_size)
            grid = utils.make_grid(samples + 0.5, nrow=f.num_components)
            utils.save_image(grid, args.checkpoints + '/samples.png')
        else:
            samples = f.sample(100, per_comp=True)
            # samples = samples.permute(1, 0, 2)
            samples = samples.reshape(-1, f.n_channels, f.image_size, f.image_size)
            grid = utils.make_grid(samples + 0.5, nrow=10)
            utils.save_image(grid, args.checkpoints + '/samples.png')  


def compute_mcc(args, config, cca_dim=20):
    isfile_rep1 = os.path.isfile(os.path.join(args.checkpoints, 'seed{}'.format(args.seed), 'test_representations.p'))
    isfile_rep2 = os.path.isfile(os.path.join(args.checkpoints, 'seed{}'.format(args.second_seed), 'test_representations.p'))

    if isfile_rep1 and isfile_rep2:
        rep1 = pickle.load(
            open(os.path.join(args.checkpoints, 'seed{}'.format(args.seed), 'test_representations.p'), 'rb'))['rep']
        rep2 = pickle.load(
            open(os.path.join(args.checkpoints, 'seed{}'.format(args.second_seed), 'test_representations.p'), 'rb'))[
            'rep']

        rep1_good = not np.isnan(rep1.sum())
        rep2_good = not np.isnan(rep2.sum())
        if rep1_good and rep2_good:
            if 'vqvae' not in config.model.architecture.lower():
                # cutoff = 50 if args.dataset == 'CIFAR100' else 5
                # ii = np.where(res_cond[0]['lab'] < cutoff)[0]  # in sample points to learn from
                # iinot = np.where(res_cond[0]['lab'] >= cutoff)[0]  # out of sample points
                cutoff = int(len(rep1)/2)  # half the test dataset
                ii = np.arange(cutoff)
                iinot = np.arange(cutoff, 2 * cutoff)

                try:
                    mcc_strong_out = mean_corr_coef_out_of_sample(x=rep1[ii], y=rep2[ii], x_test=rep1[iinot], y_test=rep2[iinot])
                    mcc_strong_in = (mean_corr_coef(x=rep1[ii], y=rep2[ii]))

                    pickle.dump({'in': mcc_strong_in, 'out': mcc_strong_out},
                                open(os.path.join(args.output, 'mcc_strong_{}_{}.p'.format(args.seed, args.second_seed)), 'wb'))
                except:
                    print('no strong mcc obtainable')
                try:
                    cca = CCA(n_components=cca_dim, max_iter=5000)
                    cca.fit(rep1[ii], rep2[ii])
                    res_out = cca.transform(rep1[iinot], rep2[iinot])
                    mcc_weak_out = mean_corr_coef(res_out[0], res_out[1])
                    res_in = cca.transform(rep1[ii], rep2[ii])
                    mcc_weak_in = mean_corr_coef(res_in[0], res_in[1])
                    print('mcc weak in: ', mcc_weak_in, ' --- ccadim = ', cca_dim)
                    print('mcc weak out: ', mcc_weak_out, ' --- ccadim = ', cca_dim)
                    pickle.dump({'in': mcc_weak_in, 'out': mcc_weak_out},
                                open(os.path.join(args.output, 'mcc_weak_{}_{}_cca_{}.p'.format(args.seed, args.second_seed, cca_dim)), 'wb'))
                except:
                    print('no weak mcc obtainable')
            else:
                for iii in range(8):
                    for jjj in range(8):
                        cutoff = int(len(rep1)/2)  # half the test dataset
                        ii = np.arange(cutoff)
                        iinot = np.arange(cutoff, 2 * cutoff)
                        cca = CCA(n_components=cca_dim, max_iter=5000)
                        cca.fit(rep1[ii,:,iii,jjj], rep2[ii,:,iii,jjj])
                        res_out = cca.transform(rep1[iinot,:,iii,jjj], rep2[iinot,:,iii,jjj])
                        mcc_weak_out = mean_corr_coef(res_out[0], res_out[1])
                        res_in = cca.transform(rep1[ii,:,iii,jjj], rep2[ii,:,iii,jjj])
                        mcc_weak_in = mean_corr_coef(res_in[0], res_in[1])
                        print(iii, jjj, ' mcc weak in: ', mcc_weak_in, ' --- ccadim = ', cca_dim)
                        print(iii, jjj, ' mcc weak out: ', mcc_weak_out, ' --- ccadim = ', cca_dim)
                        pickle.dump({'in': mcc_weak_in, 'out': mcc_weak_out},
                                    open(os.path.join(args.output, 'mcc_weak_{}_{}_cca_{}__latent_{}_{}.p'.format(args.seed, args.second_seed, cca_dim, iii, jjj)), 'wb'))


def plot_representation(args, config, cca_dim=20):
    max_seed = max_seed_baseline = args.n_sims

    if 'vqvae' not in config.model.architecture.lower():
        mcc_strong_vade_in = []
        mcc_strong_vade_out = []
        mcc_weak_vade_in = []
        mcc_weak_vade_out = []
        for i in range(args.seed, max_seed):
            for j in range(i + 1, max_seed):
                if os.path.isfile(os.path.join(args.output, 'mcc_strong_{}_{}.p'.format(i, j))):
                    temp = pickle.load(open(os.path.join(args.output, 'mcc_strong_{}_{}.p'.format(i, j)), 'rb'))
                    mcc_strong_vade_in.append(temp['in'])
                    mcc_strong_vade_out.append(temp['out'])
                if os.path.isfile(os.path.join(args.output, 'mcc_weak_{}_{}_cca_{}.p'.format(i, j, cca_dim))):                
                    temp = pickle.load(open(os.path.join(args.output, 'mcc_weak_{}_{}_cca_{}.p'.format(i, j, cca_dim)), 'rb'))
                    mcc_weak_vade_in.append(temp['in'])
                    mcc_weak_vade_out.append(temp['out'])
        mcc_strong_ivae_in = []
        mcc_strong_ivae_out = []
        mcc_weak_ivae_in = []
        mcc_weak_ivae_out = []
        for i in range(args.seed, max_seed_baseline):
            for j in range(i + 1, max_seed_baseline):
                if os.path.isfile(os.path.join(args.output_baseline, 'mcc_strong_{}_{}.p'.format(i, j))):
                    temp = pickle.load(open(os.path.join(args.output_baseline, 'mcc_strong_{}_{}.p'.format(i, j)), 'rb'))
                    mcc_strong_ivae_in.append(temp['in'])
                    mcc_strong_ivae_out.append(temp['out'])
                if os.path.isfile(os.path.join(args.output_baseline, 'mcc_weak_{}_{}_cca_{}.p'.format(i, j, cca_dim))):
                    temp = pickle.load(open(os.path.join(args.output_baseline, 'mcc_weak_{}_{}_cca_{}.p'.format(i, j, cca_dim)), 'rb'))
                    mcc_weak_ivae_in.append(temp['in'])
                    mcc_weak_ivae_out.append(temp['out'])
    else:
        mcc_weak_vade_in = []
        mcc_weak_vade_out = []
        for i in range(args.seed, max_seed):
            for j in range(i + 1, max_seed):
                for iii in range(8):
                    for jjj in range(8):
                        if os.path.isfile(os.path.join(args.output, 'mcc_weak_{}_{}_cca_{}__latent_{}_{}.p'.format(i, j, cca_dim, iii, jjj))):                
                            temp = pickle.load(open(os.path.join(args.output, 'mcc_weak_{}_{}_cca_{}__latent_{}_{}.p'.format(i, j, cca_dim, iii, jjj)), 'rb'))
                            mcc_weak_vade_in.append(temp['in'])
                            mcc_weak_vade_out.append(temp['out'])
        mcc_weak_vqst_in = []
        mcc_weak_vqst_out = []
        for i in range(args.seed, max_seed_baseline):
            for j in range(i + 1, max_seed_baseline):
                for iii in range(8):
                    for jjj in range(8):
                        if os.path.isfile(os.path.join(args.output_baseline2, 'mcc_weak_{}_{}_cca_{}__latent_{}_{}.p'.format(i, j, cca_dim, iii, jjj))):
                            temp = pickle.load(open(os.path.join(args.output_baseline2, 'mcc_weak_{}_{}_cca_{}__latent_{}_{}.p'.format(i, j, cca_dim, iii, jjj)), 'rb'))
                            mcc_weak_vqst_in.append(temp['in'])
                            mcc_weak_vqst_out.append(temp['out'])

    def _print_stats(res_vade, res_ivae, title=''):
        print('Statistics for {}:\tC\tU'.format(title))
        print('Mean:\t\t{}\t{}'.format(np.mean(res_vade), np.mean(res_ivae)))
        print('Median:\t\t{}\t{}'.format(np.median(res_vade), np.median(res_ivae)))
        print('Std:\t\t{}\t{}'.format(np.std(res_vade), np.std(res_ivae)))

    def _boxplot(res_out_vade, res_in_vade, res_out_ivae, res_in_ivae, ylabel='Linear Identifiability', ext='in', cca_dim=20):
        # sns.set_style("whitegrid")
        # sns.set_palette('deep')
        # capsprops = whiskerprops = boxprops = dict(linewidth=2)
        # medianprops = dict(linewidth=2, color='firebrick')

        sub_dfs = []

        Model = "$\mathrm{Model}$"
        raw_Model = 'Model'

        MCC = "$\mathrm{MCC}$"

        Form = ""

        data = [res_out_vade, res_in_vade, res_out_ivae, res_in_ivae]

        if 'vqvae' not in config.model.architecture.lower():
            list_of_models = ['$\mathrm{VaDE}$',
                              '$\mathrm{VaDE}$',
                              '$\mathrm{iVAE}$',
                              '$\mathrm{iVAE}$']
        else:
            list_of_models = ['$\mathrm{rVQVAE}$',
                              '$\mathrm{rVQVAE}$',
                              '$\mathrm{AE}$',
                              '$\mathrm{AE}$']           

            list_of_forms = ['$\mathrm{Out}$ $\mathrm{of}$ $\mathrm{Sample}$', '$\mathrm{In}$ $\mathrm{Sample}$',
                             '$\mathrm{Out}$ $\mathrm{of}$ $\mathrm{Sample}$', '$\mathrm{In}$ $\mathrm{Sample}$']

        for i in range(len(data)):
            local_data = data[i]
            N_vals = len(local_data)
            local_name = [list_of_models[i]] * N_vals
            local_form = [list_of_forms[i]] * N_vals
            sub_dfs.append(pd.DataFrame(list(zip(local_name, local_form, local_data)),
                                        columns =[Model, Form, MCC]))

        df = pd.concat(sub_dfs)
        print(df)

        fig, ax = plt.subplots(figsize=(5.25, 3.75))
        ax = sns.boxplot(data=df, x=Form, y=MCC, hue=Model, ax=ax, order=[list_of_forms[1], list_of_forms[0]])
        ax.legend(fontsize=14)

        fig.tight_layout()
        file_name = 'representation_'
        if config.model.final_layer:
            file_name += str(config.model.feature_size) + '_'
        plt.savefig(os.path.join(args.run, '{}{}_{}_cca_{}__.pdf'.format(file_name, args.dataset.lower(), ext, cca_dim)), bbox_inches="tight")

    if 'vqvae' not in config.model.architecture.lower():
        # print some statistics
        _print_stats(mcc_strong_vade_in, mcc_strong_ivae_in, title='strong iden. in sample')
        _print_stats(mcc_strong_vade_out, mcc_strong_ivae_out, title='strong iden. out of sample')
        _print_stats(mcc_weak_vade_in, mcc_weak_ivae_in, title='weak iden. in sample')
        _print_stats(mcc_weak_vade_out, mcc_weak_ivae_out, title='weak iden. out of sample')
        # boxplot

        _boxplot(mcc_weak_vade_out, mcc_weak_vade_in,
                 mcc_weak_ivae_out, mcc_weak_ivae_in,
                 ylabel='Linear Identifiability',
                 ext='Linear__{}_{}_{}'.format(max_seed, max_seed_baseline, config.model.architecture.lower()),
                 cca_dim=cca_dim)
        _boxplot(mcc_strong_vade_out, mcc_strong_vade_in,
                 mcc_strong_ivae_out, mcc_strong_ivae_in,
                 ylabel='Strong Identifiability',
                 ext='Strong__{}_{}_{}'.format(max_seed, max_seed_baseline, config.model.architecture.lower()),
                 cca_dim=cca_dim)
    else:
        # boxplot
        _boxplot(mcc_weak_vade_out, mcc_weak_vade_in,
                 mcc_weak_vqst_out, mcc_weak_vqst_in,
                 ylabel='Linear Identifiability',
                 ext='Linear_{}_{}_{}'.format(max_seed, max_seed_baseline, config.model.architecture.lower()),
                 cca_dim=cca_dim)  

