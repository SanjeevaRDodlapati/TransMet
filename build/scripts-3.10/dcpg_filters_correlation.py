#!/home/sdodl001/envs/TransMet_TF2161/bin/python


from __future__ import print_function
from __future__ import division

import sys
import os

import argparse

import numpy as np

from deepcpg import models as mod

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import backend as K
from deepcpg.utils import ProgressBar, to_list
from deepcpg import data as dat
import os.path as pt

class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        self.opts = opts
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Computes filter activations of a DeepCpG model')
        p.add_argument(
            'data_files',
            help='Data files',
            nargs='+')
        
        p.add_argument(
            '--model_files',
            help='Model files',
            nargs='+')
        
        p.add_argument(
            '--hinge_cells',
            help='the cells that use hinge loss',
            nargs='+')
        p.add_argument(
            '--evaluate',
            help='read data in when evaluate the model or during training',
            type=int,
            default=0)
        p.add_argument(
            '-o', '--out_file_dir',
            help='Output file',
            default='None')

        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--nb_sample',
            help='Number of samples',
            type=int)
        g.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=1024)
        g.add_argument(
            '--shuffle',
            help='Randomly sample inputs',
            action='store_true')
        
        return p

    def main(self, name, opts):
        np.random.seed(0)

        if not opts.model_files:
            raise ValueError('No model files provided!')

        K.set_learning_phase(0)
        model = mod.load_model(opts.model_files)

        weight_layer, act_layer = mod.get_first_conv_layer(model.layers, True)

        try:
            dna_idx = model.input_names.index('dna')
        except BaseException:
            raise IOError('Model is not a valid DNA model!')


        fun_outputs = to_list(act_layer.output)

        fun = K.function([to_list(model.input)[dna_idx]], fun_outputs)

       


        output_names = model.output_names
        #print("output_names(shall be None):", output_names)
        data_reader = mod.DataReader(
            output_names=output_names,
            use_dna=True,
            dna_wlen=to_list(model.input_shape)[dna_idx][1],
            # hinge_cells=opts.hinge_cells,
            # evaluate = opts.evaluate
        )
        #nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        nb_sample = dat.get_nb_sample(opts.data_files)
        data_reader = data_reader(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False,
                                  #shuffle=opts.shuffle
                                  )


        # n = 991
        center = 991 // 2 + 1
        L = [1 / i for i in range(center, 1, -1)]
        R = [1 / i for i in range(2, center + 1)]
        w = L + [1] + R
        w = np.array(w)
        w = w.reshape((991, 1))
        acts = []

        for data in data_reader:
            if isinstance(data, tuple):
                inputs, outputs, weights = data
            else:
                inputs = data
            if isinstance(inputs, dict):
                inputs = list(inputs.values())

            fun_eval = fun(inputs)
            act = fun_eval[0]
            #get the average of act for each filter

            #print("act:", act)
            # print("act.shape:", act.shape) #(256, 991, 128)
            # print("preds.shape", preds.shape) #(10, 256, 1)
            # print(model.input_names)

            weighted_f_act = np.multiply(act, w)
            weighted_mean_f_act = np.mean(weighted_f_act, axis=1)

            act = np.mean(act, axis = 0)

            acts.append(act)

        acts = np.stack(acts) #Nx991x128
        acts = np.mean(acts, axis = 0) #991 x 128

        np.save(pt.join(opts.out_file_dir, 'acts.npy'), acts)

        # corr = np.corrcoef(acts, rowvar=False)
        # corr = pd.DataFrame(corr, 
        #                     columns = [str(i) for i in range(128)], 
        #                     index = [str(i) for i in range(128)])

        # corr.dropna(axis=1, how='all', inplace=True)
        # corr.dropna(axis=0, how='all',  inplace=True)
        




        # fig, ax = plt.subplots(figsize=(9,7))
        # ax = sns.heatmap(corr,
        #                  #linewidth=.5, 
        #                  cmap='coolwarm',
        #                  robust=True
        #                 )
        # plt.show()

        return 0


if __name__ == '__main__':
    app = App()
    app.run(sys.argv)
