from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy

from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt

from data_objects.inter_result_tracker import InterResultTracker
from deeprobust.image.config import defense_params, attack_params
from deeprobust.image import attack as Attack
from deeprobust.image import defense as Defense

import sparselearning

from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, \
    ResNet18, ResNet50, ResNet101
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, \
    get_tinyimagenet_dataloaders

import numpy as np
import warnings
import csv

from sklearn import metrics

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {
    'MLPCIFAR10': (MLP_CIFAR10, []),
    'lenet5': (LeNet_5_Caffe, []),
    'lenet300-100': (LeNet_300_100, []),
    'ResNet101': (()),
    'ResNet50': (()),
    'ResNet34': (()),
    'ResNet18': (()),
    'alexnet-s': (AlexNet, ['s', 10]),
    'alexnet-b': (AlexNet, ['b', 10]),
    'vgg-c': (VGG16, ['C', 10]),
    'vgg-d': (VGG16, ['D', 10]),
    'vgg-like': (VGG16, ['like', 10]),
    'wrn-28-2': (WideResNet, [28, 2, 10, 0.3]),
    'wrn-22-8': (WideResNet, [22, 8, 10, 0.3]),
    'wrn-16-8': (WideResNet, [16, 8, 10, 0.3]),
    'wrn-16-10': (WideResNet, [16, 10, 10, 0.3])}

attacks = {
    'PGD': Attack.pgd.PGD,
    'cw': Attack.cw.CarliniWagner,
    'FGSM': Attack.fgsm.FGSM,
    'LBFGS': Attack.lbfgs.LBFGS,
    'DeepFool': Attack.deepfool.DeepFool,
    'Onepixel': Attack.onepixel.Onepixel
}

defences = {
    'FGSM': Defense.fgsmtraining.FGSMtraining,
    'PGD': Defense.pgdtraining.PGDtraining,
    'YOPOPGD': Defense.YOPO.YOPOpgd.FASTPGD,
}


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density,
                                               hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def save(epoch, state_dict, optimizer, location):
    print(f'save in {location}')
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': {'defaults': optimizer.defaults, 'param_groups': optimizer.param_groups, 'state': optimizer.state},
    }, location)


def train(args, model, device, train_loader, optimizer, epoch, mask=None):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                       100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))

    # training summary
    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary',
        train_loss / batch_idx, correct, n, 100. * correct / float(n)))
    train_accuracy = correct / float(n)
    return train_loss, train_accuracy


def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    print_and_log(f'float n: {float(n)}')
    return correct / float(n), test_loss


def adversarial_training(model, method_name, train_loader, test_loader, defence_config, evaluation_pack):
    print(f'Start adversarial training with training method: {method_name}.')
    method = defences[method_name]
    model = method(model, 'cuda')
    model = model.generate(train_loader, test_loader, evaluation_pack, **defense_params[defence_config])
    return model


def adversarial_attack(args, model, method_name, test_loader, attack_config):
    print(f'Start adversarial attack with attack method: {method_name}.')
    total = 0
    correct_orig = 0
    correct_adv = 0
    recall_orig = 0
    precision_orig = 0
    recall_adv = 0
    precision_adv = 0
    f1_orig = 0
    f1_adv = 0
    method = attacks[method_name]
    for batch_num, (data, target) in enumerate(test_loader):
        print(f'Adversarial attack batch {batch_num}/{len(test_loader)}')
        total += len(data)
        data = data.to('cuda').float()

        predict0 = model(data)
        predict0 = predict0.argmax(dim=1, keepdim=True)

        adversary = method(model)
        AdvExArray = adversary.generate(data, target, **attack_params[attack_config]).float()

        predict1 = model(AdvExArray)
        predict1 = predict1.argmax(dim=1, keepdim=True)

        labels = np.array(target.cpu())
        pred_orig = np.array(predict0.cpu()).flatten()
        pred_adv = np.array(predict1.cpu()).flatten()
        correct_orig += np.sum(labels == pred_orig)
        correct_adv += np.sum(labels == pred_adv)

        recall_orig += metrics.recall_score(labels, pred_orig, average='micro')
        recall_adv += metrics.recall_score(labels, pred_adv, average='micro')

        precision_orig += metrics.precision_score(labels, pred_orig, average='micro')
        precision_adv += metrics.precision_score(labels, pred_adv, average='micro')

        f1_orig += metrics.f1_score(labels, pred_orig, average='micro')
        f1_adv += metrics.f1_score(labels, pred_adv, average='micro')
    print('=== Results ===')
    print(f'Total: {total}')
    print(f'Original predictions: {correct_orig}/{total} ({100 * correct_orig / total}%)')
    print(f'Adversarial predictions: {correct_adv}/{total} ({100 * correct_adv / total}%)')
    f = lambda x: np.mean(x) / batch_num
    row_to_file(args, [f'{100 * correct_orig / total}%', f'{100 * correct_adv / total}%', f(recall_orig), f(recall_adv),
                       f(pred_orig), f(pred_adv), f(f1_orig), f(f1_adv)])
    return [f'{100 * correct_orig / total}%', f'{100 * correct_adv / total}%', f(recall_orig), f(recall_adv),
            f(pred_orig), f(pred_adv), f(f1_orig), f(f1_adv)]


def row_to_file(args, values):
    values = [args.identifier, args.batch_size, args.epochs, args.momentum, args.lr, args.save, args.save_adv,
              args.data, args.resume,
              args.model, args.train_reg, args.train_adv, args.adv_attack, args.sparse, args.growth, args.death,
              args.death_rate] \
             + values
    with open(args.result_file, 'a') as f:
        writer = csv.writer(f)
        print(values)
        writer.writerow(values)


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
    return mycmap


def track(tracker, device, args, epoch, train_accuracy, train_loss, model, valid_loader):
    val_acc, val_loss = evaluate(args, model, device, valid_loader)
    tracker.add('epoch', epoch)
    tracker.add('train_accuracy', train_accuracy)
    tracker.add('train_loss', train_loss)
    tracker.add('val_acc', val_acc)
    tracker.add('val_loss', val_loss)
    attack_config = f'{args.adv_attack}_{args.data}'
    tracker.add('adv_attack_val', adversarial_attack(args, model, args.adv_attack, valid_loader, attack_config))
    tracker.add('model_size', np.sum([np.count_nonzero(p.cpu().detach().numpy())
                                      for p in model.parameters()]))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=f'models/{randomhash}.pt',
                        help='path to save the final model')
    parser.add_argument('--save_adv', type=str, default=f'models/{randomhash}_adv.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1,
                        help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true',
                        help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true',
                        help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--train_reg', action='store_true', help='Whether model should be trained regularly.')
    parser.add_argument('--train_adv', type=str, help='Which adversarial train method to use. Null '
                                                      ' for no adversarial training.')
    parser.add_argument('--adv_attack', type=str,
                        help='Which adversarial attack method to use. Null for no adversarial attack.')
    parser.add_argument('--print_model_size', action='store_true',
                        help='Whether to print the model size after loading.')
    parser.add_argument('--visualise', action='store_true',
                        help='Whether to visualise the model.')
    parser.add_argument('--result_file', type=str, default='results.csv')

    parser.add_argument('--identifier', type=str, default=time.time(), help='Used to identify run')
    parser.add_argument('--track', action='store_true', help='whether to track certain features')
    parser.add_argument('--tracker_loc', type=str)
    parser.add_argument('--track_interval', type=int, default=10)
    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if not args.tracker_loc:
        args.tracker_loc = f'results/tracker_files/{args.save_adv[7:-3]}_{args.identifier}.pt'
    tracker = InterResultTracker(args.tracker_loc, args=args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('=' * 80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))
        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        elif args.data == 'CIFAR10':
            c = 10
            stride = 4
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                              max_threads=args.max_threads)
        elif args.data == 'CIFAR100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split,
                                                                               max_threads=args.max_threads)
            c = 100
            stride = 4
        elif args.data == 'tiny_imagenet':
            train_loader, valid_loader, test_loader = get_tinyimagenet_dataloaders(args, args.valid_split)
            c = 200
            stride = 28
        else:
            raise Exception(f'Dataset name not recognized: {args.data}')
        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        elif args.model == 'ResNet18':
            model = ResNet18(c=c).to(device)
        elif args.model == 'ResNet34':
            model = ResNet34(c=c, stride=stride).to(device)
        elif args.model == 'ResNet50':
            model = ResNet50(c=c).to(device)
        elif args.model == 'ResNet101':
            model = ResNet101(c=c).to(device)
        else:
            cls, cls_args = models[args.model]
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)

        print_and_log('=' * 60)
        print_and_log('Prune mode: {0}'.format(args.death))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2,
                                  nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(args.epochs / 2) * args.multiplier,
                                                                        int(args.epochs * 3 / 4) * args.multiplier],
                                                            last_epoch=-1)

        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                if args.train_reg:
                    args.start_epoch = checkpoint['epoch']
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    model.load_state_dict(checkpoint['state_dict'])
                    print_and_log("=> loaded checkpoint '{}' (epoch {})"
                                  .format(args.resume, checkpoint['epoch']))
                else:
                    print_and_log("=> loaded checkpoint '{}'"
                                  .format(args.resume))
                    model.load_state_dict(checkpoint['state_dict'])
                # print_and_log('Testing...')
                # evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                # plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale=None,
                                       dynamic_loss_scale=True,
                                       dynamic_loss_args={'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        if args.sparse:
            decay = CosineDecay(args.death_rate, len(train_loader) * (args.epochs * args.multiplier))
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay,
                           growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        best_acc = 0.0
        epoch = 0
        print(f'Train: {args.train_reg}')
        if args.train_reg:
            print(f'Start epoch: {args.start_epoch}')
            for epoch in range(args.start_epoch, args.epochs * args.multiplier + 1):
                # if epoch % 10 == 0:
                #     args.lr = args.lr * 0.1
                t0 = time.time()
                train_accuracy, train_loss = train(args, model, device, train_loader, optimizer, epoch, mask)
                lr_scheduler.step()
                if args.valid_split > 0.0:
                    if args.track and (epoch % args.track_interval == 0 or epoch == 0):
                        track(tracker, device, args, epoch, train_accuracy, train_loss, model, valid_loader)
                # if val_acc > best_acc:
                #     print('Saving model')
                #     best_acc = val_acc
                #     # torch.save(model.state_dict(), args.save)
                #     save(epoch, model.state_dict(), optimizer, args.save)
            save(epoch, model.state_dict(), optimizer, args.save)

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
                optimizer.param_groups[0]['lr'], time.time() - t0))
            print('Testing model')
            model.load_state_dict(torch.load(args.save)['state_dict'])
            tracker.add('clean_test_acc', evaluate(args, model, device, test_loader, is_test_set=True))
            print_and_log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))
            if args.sparse:
                layer_fired_weights, total_fired_weights = mask.fired_masks_update()
                for name in layer_fired_weights:
                    print('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
                print('The final percentage of the total fired weights is:', total_fired_weights)

        if args.print_model_size:
            print_and_log(np.sum([np.count_nonzero(p.cpu().
                                                   detach().numpy()) for p in model.parameters()]))

        if args.train_adv:
            defense_config = f'{args.train_adv}_{args.data}'
            model = adversarial_training(model, args.train_adv, train_loader, test_loader, defense_config,
                                         {'track': track, 'args': args, 'valid_loader': valid_loader,
                                          'track_interval': args.track_interval,
                                          'tracker': tracker})
            save(epoch, model.state_dict(), optimizer, args.save_adv)

        if args.print_model_size:
            print_and_log(np.sum([np.count_nonzero(p.cpu().
                                                   detach().numpy()) for p in model.parameters()]))

        if args.adv_attack:
            attack_config = f'{args.adv_attack}_{args.data}'
            tracker.add('adv_test_acc', adversarial_attack(args, model, args.adv_attack, test_loader, attack_config))

        if args.visualise:
            for data, target in train_loader:
                # Send the data and label to the device

                # Set requires_grad attribute of tensor. Important for Attack
                data.requires_grad = True

                # Forward pass the data through the model
                output = model(data)
                init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                # If the initial prediction is wrong, dont bother attacking, just move on
                if init_pred.item() != target.item():
                    continue


if __name__ == '__main__':
    main()
