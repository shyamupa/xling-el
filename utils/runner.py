# from torchnet.engine import Engine
import sys, shutil
import os
import torch
import logging

from torch.nn import Parameter

__author__ = 'Shyam'


class MyEngine(object):
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer):
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'], state)
                    state['output'] = output
                    state['loss'] = loss
                    loss.backward()
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator):
        state = {
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_end', state)
        return state


class Runner(MyEngine):
    def __init__(self, model, optimizer):
        super(Runner, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.hooks['on_sample'] = self.on_sample
        self.hooks['on_forward'] = self.on_forward
        self.hooks['on_start_epoch'] = self.on_start_epoch
        self.hooks['on_end_epoch'] = self.on_end_epoch
        self.hooks['on_update'] = self.on_update
        self.hooks['on_end'] = self.on_end

    def run(self, maxepoch=1, train_it=None):
        """
        :param maxepoch: max epochs
        :param train_it: iterator that returns train samples
        :return:
        """
        self.train(self.handle_sample, train_it, maxepoch=maxepoch, optimizer=self.optimizer)

    def on_sample(self, state):
        raise NotImplementedError

    def handle_sample(self, sample, state):
        raise NotImplementedError

    def on_forward(self, state):
        raise NotImplementedError

    def on_start_epoch(self, state):
        raise NotImplementedError

    def on_end_epoch(self, state):
        raise NotImplementedError

    def on_end(self, state):
        raise NotImplementedError

    def on_update(self, state):
        raise NotImplementedError

    @staticmethod
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        """
        From https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3

        Example usage

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

        :param state:
        :param is_best:
        :param filename:
        :return:
        """
        logging.info("saving model to %s", filename)
        torch.save(state, filename)
        if is_best:
            logging.info("copying to best ...")
            shutil.copyfile(filename, filename + '_best.pth.tar')



    @staticmethod
    def load_checkpoint(model, optimizer, ckpt_path):
        if os.path.isfile(ckpt_path):
            logging.info("=> loading checkpoint %s", ckpt_path)
            checkpoint = torch.load(ckpt_path,map_location='cpu')
            # model.load_state_dict(checkpoint['state_dict'])
            # if model does not have params, its ok.
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # any other relevant state variables can be extracted from the checkpoint dict
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(ckpt_path, checkpoint['epoch']))
            logging.info("=> loaded checkpoint!")
            return checkpoint
        else:
            logging.info("=> no checkpoint at %s !!!", ckpt_path)
            logging.info("dying ...")
            sys.exit(0)
