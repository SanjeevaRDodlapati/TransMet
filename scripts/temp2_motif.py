from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as pt

import argparse
import h5py as h5
from tensorflow.keras import backend as K
import numpy as np
import logging
import six

from deepcpg import data as dat
from deepcpg import models as mod
from deepcpg.data import hdf, dna
from deepcpg.utils import ProgressBar, to_list, linear_weights

from deepcpg.utils import EPS, linear_weights, make_dir

from collections import OrderedDict
import subprocess
#from deepcpg.motifs import read_meme_db, get_report, read_meme_from_other_model
from deepcpg.motifs import read_meme_db, get_report

import pandas as pd

import time
import concurrent.futures
from concurrent.futures import as_completed

import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from multiprocessing import Pool
import tqdm


WEBLOGO_OPTS = '-X NO -Y NO --errorbars NO --fineprint ""'
WEBLOGO_OPTS += ' --logo-font Arial-BoldMT'
WEBLOGO_OPTS += ' -C "#CB2026" A A'
WEBLOGO_OPTS += ' -C "#34459C" C C'
WEBLOGO_OPTS += ' -C "#FBB116" G G'
WEBLOGO_OPTS += ' -C "#0C8040" T T'

ALPHABET = dna.get_alphabet(False)
MEME_ALPHABET = OrderedDict([('A', 0), ('C', 1), ('G', 2), ('T', 3)])
ALPHABET_R = OrderedDict([(value, key) for key, value in ALPHABET.items()])


#global variables for multithreading
pwms = np.zeros((11,4,128)) #result pwms matrix
background_freq = {0:0, 1:0, 2:0, 3:0}
nb_sites = np.array([0]*128)
seqs = []
act = []
thrs_avg = []

def zeropad_array(x, n, axis=0):
    pad_shape = list(x.shape)
    pad_shape[axis] += 2 * n
    pad = np.zeros(pad_shape, dtype=x.dtype)
    idx = [slice(0, x.shape[i]) for i in range(x.ndim)]
    idx[axis] = slice(n, n + x.shape[axis])
    pad[idx] = x
    return pad

def ranges_to_list(x, start=0, stop=None):
    s = set()
    for xi in x:
        xi = str(xi)
        if xi.find('-') >= 0:
            t = xi.split('-')
            if len(t) != 2:
                raise ValueError('Invalid range!')
            if len(t[0]) == 0:
                t[0] = start
            if len(t[1]) == 0:
                t[1] = stop
            s |= set(range(int(t[0]), int(t[1]) + 1))
        else:
            s.add(int(xi))
    s = sorted(list(s))
    return s

def format_out_of(out, of):
    return '%.1f%% (%d / %d)' % (out / of * 100, out, of)

def idx_gen(pwms, act, thrs_avg, seqs, filter_del, filter_len):
    for f in range(pwms.shape[-1]):
        idx = np.nonzero(np.greater(act[:, :, f], thrs_avg[f]))
        yield idx, seqs, filter_del, filter_len, f

def kmer_filter(argsumets):
    idx, seqs, filter_del, filter_len, f = argsumets
    # print(f'filter: {f}')
    # kmers = []
    kmers = np.zeros((idx[0].shape[0], 11, 4))
    for k in range(len(idx[0])):
        i = int(idx[0][k])
        j = int(idx[1][k])
        # # should this condition be seqs.shape[1] - filter_del - 1 ????
        if j < filter_del or j > (seqs.shape[1] - filter_len - 1):
            continue
        kmer = seqs[i, (j - filter_del):(j + filter_del + filter_len % 2)]

        # use not cut dna sequence
        # kmer = seqs[i, j:j+11]

        # kmers.append(kmer)
        kmers[k] = kmer
    # kmers = np.ndarray(kmers)
    nb_kmers = kmers.shape[0]
    # print(f'filter: {f}, nb_sites: {nb_kmers}')
    return (f, kmers, nb_kmers)

def get_act_kmers(filter_act, filter_len, seqs, thr_per=0.5, thr_max=25000,
                  log=None):
    assert filter_act.shape[0] == seqs.shape[0]
    assert filter_act.shape[1] == seqs.shape[1]

    _thr_per = 0
    if thr_per:
        filter_act_mean = filter_act.mean()
        filter_act_norm = filter_act - filter_act_mean
        _thr_per = thr_per * filter_act_norm.max() + filter_act_mean

        if log:
            tmp = format_out_of(np.sum(filter_act >= _thr_per), filter_act.size)
            log('%s passed percentage threshold' % tmp)

    _thr_max = 0
    if thr_max:
        thr_max = min(thr_max, filter_act.size)
        _thr_max = np.percentile(filter_act,
                                 (1 - thr_max / filter_act.size) * 100)
        if log:
            tmp = format_out_of(np.sum(filter_act >= _thr_max), filter_act.size)
            log('%s passed maximum threshold' % tmp)

    kmers = []
    thr = max(_thr_per, _thr_max)
    idx = np.nonzero(filter_act >= thr)
    filter_del = filter_len // 2
    for k in range(len(idx[0])):
        i = int(idx[0][k])
        j = int(idx[1][k])
        if j < filter_del or j > (seqs.shape[1] - filter_len - 1):
            continue
        kmer = seqs[i, (j - filter_del):(j + filter_del + filter_len % 2)]
        kmers.append(kmer)
    kmers = np.array(kmers)

    return kmers


def write_kmers(kmers, filename):
    char_kmers = np.chararray(kmers.shape)
    for _char, _int in six.iteritems(ALPHABET):
        char_kmers[kmers == _int] = _char

    with open(filename, 'a') as fh:
        for i, kmer in enumerate(char_kmers):
            print('>%d' % i, file=fh)
            print(kmer.tobytes().decode(), file=fh)


def plot_filter_densities(densities, filename=None):
    sns.set(font_scale=1.3)
    fig, ax = plt.subplots()
    sns.histplot(densities, kde=False, ax=ax)
    ax.set_xlabel('Activation')
    if filename:
        fig.savefig(filename)
        plt.close()

def plot_filter_heatmap(weights, filename=None):
    param_range = abs(weights).max()

    fig, ax = plt.subplots(figsize=(weights.shape[1], weights.shape[0]))
    sns.heatmap(weights, cmap='RdYlBu_r', linewidths=0.2, vmin=-param_range,
                vmax=param_range, ax=ax)
    ax.set_xticklabels(range(1, weights.shape[1] + 1))
    labels = [ALPHABET_R[i] for i in reversed(range(weights.shape[0]))]
    ax.set_yticklabels(labels, rotation='horizontal', size=10)
    if filename:
        plt.savefig(filename)
        plt.close()

def plot_pca(act, pc_x=1, pc_y=2, labels=None, filename=None):
    act = act.T
    pca = PCA()
    pca.fit(act)
    eig_vec = pca.transform(act)
    data = pd.DataFrame(eig_vec)
    data.columns = ['PC%d' % i for i in range(data.shape[1])]
    data['act_mean'] = act.mean(axis=1)

    pc_x = 'PC%d' % pc_x
    pc_y = 'PC%d' % pc_y
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(data[pc_x], data[pc_y],
                         c=data['act_mean'], cmap='RdBu_r')
    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    fig.colorbar(scatter)
    if labels:
        for i, row in data.iterrows():
            ax.annotate('%d' % labels[i], xy=(row[pc_x], row[pc_y]),
                        fontsize=10)
    if filename:
        fig.savefig(filename)
        plt.close()

def map_alphabets(values, src_alphabet, dst_alphabet):
    assert len(src_alphabet) == len(dst_alphabet)
    _values = values.copy()
    for src_char, src_int in six.iteritems(src_alphabet):
        _values[dst_alphabet[src_char]] = values[src_int]
    return _values

def open_meme(filename, background_freq):

    # open file for writing
    meme_file = open(filename, 'w')

    # print intro material
    print('MEME version 4', file=meme_file)
    print('', file=meme_file)
    print('ALPHABET= %s' % ''.join(list(MEME_ALPHABET.keys())), file=meme_file)
    print('', file=meme_file)
    print('Background letter frequencies:', file=meme_file)
    nt_freq_str = []
    for nt_char, nt_int in six.iteritems(MEME_ALPHABET):
        nt_freq_str.append('%s %.4f' % (nt_char, background_freq[nt_int]))
    print(' '.join(nt_freq_str), file=meme_file)
    print('', file=meme_file)

    return meme_file

def info_content(pwm):
    pwm = np.atleast_2d(pwm)
    return np.sum(pwm * np.log2(pwm + EPS) + 0.5)


def add_to_meme(meme_file, idx, pwm, nb_site, trim_thr=None):
    if trim_thr:
        start = 0
        while start < pwm.shape[0] and \
                info_content(pwm[start]) < trim_thr:
            start += 1

        end = len(pwm) - 1
        while end >= 0 and \
                info_content(pwm[end]) < trim_thr:
            end -= 1
        if start > end:
            return
        pwm = pwm[start:end]

    pwm = map_alphabets(pwm.T, ALPHABET, MEME_ALPHABET).T

    print('MOTIF filter%d' % idx, file=meme_file)
    tmp = 'letter-probability matrix: alength= %d w= %d nsites= %d'
    tmp = tmp % (len(MEME_ALPHABET), len(pwm), nb_site)
    print(tmp, file=meme_file)

    for row in pwm:
        row = ' '.join(['%.4f' % freq for freq in row])
        print(row, file=meme_file)
    print('', file=meme_file)



def plot_logo(fasta_file, out_file, out_format=None, options=''):
    if out_format is None:
        out_format = pt.splitext(out_file)[1][1:]
    cmd = 'weblogo {opts} -s large < {inp} > {out} -F {f} 2> /dev/null'
    cmd = cmd.format(opts=options, inp=fasta_file, out=out_file,
                    f=out_format)
    # print(f'fa_file: {fasta_file}, cmd: {cmd}, out_file: {out_file}')
    subprocess.call(cmd, shell=True)
    
    # cmd = 'weblogo {opts} -s large < {inp} > {out}'
    # cmd = cmd.format(opts=options, inp=fasta_file, out=out_file)
    #
    # subprocess.run(cmd, capture_output = True, shell = True, check = True)

def get_motif_from_weights(weights):
    idx = weights.argmax(axis=0)
    return ''.join([ALPHABET_R[i] for i in idx])

def filters(start_idx):
    global act
    global thrs_avg
    global seqs
    global nb_sites
    global pwms
    
    for f in range(start_idx, start_idx+16):
        idx = np.nonzero(np.greater(act[:,:,f],thrs_avg[f]))

        kmers = []
        for k in range(len(idx[0])):
            i = int(idx[0][k])
            j = int(idx[1][k])
            kmer = seqs[i, j:j+11]
                    
            kmers.append(kmer)

        nb_sites[f] += len(kmers)
                
        if kmers:
            pwms[:,:,f] += np.sum(dna.int_to_onehot(np.array(kmers)), axis = 0)

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
        # p.add_argument(
        #     '-o', '--out_file',
        #     help='Output file',
        #     default='activations.h5')
        p.add_argument(
            '-o', '--out_dir',
            help='Output directory',
            default='.')

        g = p.add_argument_group('arguments for summarizing activations')
        g.add_argument(
            '--act_fun',
            help='Function for summarizing activations in each sequence window',
            choices=['mean', 'wmean', 'max'])
        g.add_argument(
            '--act_wlen',
            help='Maximal length of sequence windows',
            type=int)

        g = p.add_argument_group('output arguments')
        g.add_argument(
            '--store_outputs',
            help='Store output labels',
            action='store_true')
        g.add_argument(
            '--store_preds',
            help='Store model predictions',
            action='store_true')
        g.add_argument(
            '--store_inputs',
            help='Store model inputs',
            action='store_true')

        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--nb_sample',
            help='Number of samples',
            type=int)
        g.add_argument(
            '--shuffle',
            help='Randomly sample inputs',
            action='store_true')
        g.add_argument(
            '--batch_size',
            help='Batch size',
            type=int,
            default=64)


        p.add_argument(
            '--hinge_cells',
            help='the cells that use hinge loss',
            nargs='+')
        p.add_argument(
            '--evaluate',
            help='read data in when evaluate the model or during training',
            type=int,
            default=0)

        # g.add_argument(
        #     '--compare_models',
        #     help='indicate whether we compare between models or not',
        #     type=int,
        #     default=0)

        g = p.add_argument_group('motif arguments')
        g.add_argument(
            '--act_thr_per',
            help='Minimum activation threshold of aligned sequence fragments.'
            ' Percentage of maximum activation above the mean activation.',
            default=0.5,
            type=float)
        g.add_argument(
            '--act_thr_max',
            help='Maximum number of aligned sequence fragments',
            type=int,
            default=25000)
        g.add_argument(
            '--out_format',
            help='Output format of motif logos and plots',
            default='pdf')
        g.add_argument(
            '--weblogo_opts',
            help='Command line options of Weblogo command',
            default=WEBLOGO_OPTS)
        g.add_argument(
            '--delete_fasta',
            help='Delete fasta files after visualizing motif to reduce disk'
            ' storage',
            action='store_true')
        g.add_argument(
            '-m', '--motif_dbs',
            help='MEME databases for motif comparison',
            nargs='+')
        
        g.add_argument(
            '--fdr',
            help='FDR for motif comparision',
            default=1,
            type=float)
        g = p.add_argument_group('motif analysis')
        g.add_argument(
            '--plot_dens',
            help='Plot filter activation density',
            action='store_true')
        g.add_argument(
            '--plot_heat',
            help='Plot filter heatmaps',
            action='store_true')
        g.add_argument(
            '--plot_pca',
            help='Plot first two principal componets of motif activities',
            action='store_true')
        g.add_argument(
            '--nb_sample_pca',
            help='Number of samples in PCA matrix',
            type=int,
            default=1000)
        
        g = p.add_argument_group('advanced arguments')
        g.add_argument(
            '--trim_thr',
            help='Threshold from trimming uninformative sites of PWM',
            type=float)
        g.add_argument(
            '--filters',
            help='Indicies of filters (starting from 0) to be selected. Can be'
            ' range of filters, e.g. -10 50-60 to select filter 0-10 and'
            ' 50-50.',
            nargs='+')
        # g.add_argument(
        #     '--nb_sample',
        #     help='Maximum number of input samples',
        #     type=int)
        g.add_argument(
            '--seed',
            help='Seed of random number generator',
            type=int,
            default=0)
        g.add_argument(
            '--verbose',
            help='More detailed log messages',
            action='store_true')
        g.add_argument(
            '--log_file',
            help='Write log messages to file')

        return p
    
    def plot_filename(self, dirname, basename, out_format=None):
        if out_format is None:
            out_format = self.opts.out_format
        return pt.join(dirname, '%s.%s' % (basename, out_format))

    
    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file,
                            format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        if opts.verbose:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
            log.debug(opts)


        nb_filter = 128
        filters_idx = range(nb_filter)
        

        if not opts.model_files:
            raise ValueError('No model files provided!')

        log.info('Loading model ...')
        K.set_learning_phase(0)
        model = mod.load_model(opts.model_files, log=log.info)

        weight_layer, act_layer = mod.get_first_conv_layer(model.layers, True)
        log.info('Using activation layer "%s"' % act_layer.name)
        log.info('Using weight layer "%s"' % weight_layer.name)

        weights = weight_layer.get_weights()
        filters_weights = weights[0]

        if filters_weights.ndim == 4:
            # For backward compatibility, support filter weights of shape
            # [filter_len, 1, nb_input_features, nb_output_features]
            assert filters_weights.shape[1] == 1
            filters_weights = filters_weights[:, 0, :]
        assert filters_weights.shape[1] == 4

        try:
            dna_idx = model.input_names.index('dna')
        except BaseException:
            raise IOError('Model is not a valid DNA model!')

        fun_outputs = to_list(act_layer.output)
        if opts.store_preds:
            fun_outputs += to_list(model.output)
        fun = K.function([to_list(model.input)[dna_idx]], fun_outputs)


        make_dir(opts.out_dir)
        sub_dirs = dict()
        names = ['logos', 'fa']
        if opts.plot_dens:
            names.append('dens')
        if opts.plot_heat:
            names.append('heat')
        if opts.motif_dbs:
            names.append('tomtom')
        for name in names:
            dirname = pt.join(opts.out_dir, name)
            sub_dirs[name] = dirname
            make_dir(dirname)


        log.info('Reading data ...')

        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        seeds = range(1)
        if nb_sample > 100000:
            nb_sample = 100000
            seeds = range(10)
        log.info('Number of seeds to sample data : "%s"' % str(len(seeds)))


        thrs = []
        for seed in seeds:
            np.random.seed(seed)
            output_names = model.output_names
        
            data_reader = mod.DataReader(
                output_names=output_names,
                use_dna=True,
                dna_wlen=to_list(model.input_shape)[dna_idx][1],
                # hinge_cells=opts.hinge_cells,
                # evaluate = opts.evaluate
            )

            data_reader = data_reader(opts.data_files,
                                     nb_sample=nb_sample,
                                     batch_size=opts.batch_size,
                                     loop=False,
                                     shuffle=opts.shuffle)


            log.info('Computing activations with seed: "%s"' % str(seed+1))
            progbar = ProgressBar(nb_sample, log.info)
 

            acts = []
            for data in data_reader:
                if isinstance(data, tuple):
                    inputs, _ , _ = data
                else:
                    inputs = data
                if isinstance(inputs, dict):
                    inputs = list(inputs.values())
                batch_size = len(inputs[0])
                progbar.update(batch_size)

                fun_eval = fun(inputs)
                act = fun_eval[0]
                acts.append(act)
            
            acts_concat = np.concatenate(acts, axis = 0)

            acts_mean = np.mean(acts_concat, axis = (0,1))
            acts_norm = np.subtract(acts_concat, acts_mean)

            thr_per = 0.5

            _thr_per = 0
            _thr_per = thr_per * np.amax(acts_norm, axis = (0,1)) + acts_mean


            # thr_max = np.array([25000] * acts_concat.shape[-1])

            # s = acts_concat.shape[0] * acts_concat.shape[1]
            # thr_max = np.minimum(thr_max, s)

            # _thr_max = np.zeros(acts_concat.shape[-1])

            # for i in range(acts_concat.shape[-1]):
            #     _thr_max[i] = np.percentile(acts_concat[:,:,i],
            #                      (1 - thr_max[i] / s) * 100,
            #                      axis = (0,1))

            # thr = np.maximum(_thr_per, _thr_max)
            thr = _thr_per
            thrs.append(thr)

        
        #get the average threshold over 5 times of activation calculation
        thrs_avg = np.mean(np.array(thrs), axis = 0)

        log.info("The final thresh for each filters are: {}".format(' '.join(map(str, thrs_avg))))

        #use all data to calculate the activation now
        log.info('Reading all data now...')
        
        #output_names = model.output_names


        data_reader2 = mod.DataReader(
            output_names=output_names,
            use_dna=True,
            dna_wlen=to_list(model.input_shape)[dna_idx][1],
            # hinge_cells=opts.hinge_cells,
            # evaluate = opts.evaluate
        )
        #change the nb sample to all data
        nb_sample = dat.get_nb_sample(opts.data_files)
        #nb_sample = dat.get_nb_sample(opts.data_files,10000)

        data_reader2 = data_reader2(opts.data_files,
                                  nb_sample=nb_sample,
                                  batch_size=opts.batch_size,
                                  loop=False,
                                  shuffle=False)
        


        # pwms = np.ones((11,4,128)) #result pwms matrix

        # background_freq = {0:0, 1:0, 2:0, 3:0}
        # nb_sites = np.array([0]*128)

        filter_len = 11
        filter_del = filter_len // 2
        #batch_size = len(inputs[0])


        progbar = ProgressBar(nb_sample, log.info)


        for data in data_reader2:
            if isinstance(data, tuple):
                inputs, _ , _ = data
            else:
                inputs = data
            if isinstance(inputs, dict):
                inputs = list(inputs.values())
            
            batch_size = len(inputs[0])
            progbar.update(batch_size)


            fun_eval = fun(inputs)

            act = fun_eval[0]

            filters_act = act

            seqs = inputs[0]
            if seqs.shape[1] != filters_act.shape[1]:
                # Trim sequence length to length of activation layer
                tmp = (seqs.shape[1] - filters_act.shape[1]) // 2
                seqs = seqs[:, tmp:(tmp + filters_act.shape[1])]
                assert seqs.shape[1] == filters_act.shape[1]

            for key in background_freq.keys():
                background_freq[key] += np.sum(seqs == key)

            pool = Pool(int(os.getenv('SLURM_CPUS_PER_TASK', 1)))
            pooled = pool.map(kmer_filter, idx_gen(pwms, act, thrs_avg, seqs, filter_del, filter_len))
            pooled = list(pooled)
            kmers = [pooled[i][1] for i in range(len(pooled))]
            nb_kmers = [pooled[i][2] for i in range(len(pooled))]
            # kmers = np.array(kmers)
            # print(f'kmers shape: {kmers.shape}')


            for f in range(pwms.shape[-1]):
                pwms[:, :, f] += np.sum(kmers[f], axis=0)
                nb_sites[f] += nb_kmers[f]

                logo_file = pt.join(sub_dirs['fa'], '%03d.fa' % f)
                write_kmers(kmers[f], logo_file)



        total = sum(background_freq.values())
        for key in background_freq.keys():
            background_freq[key] /= total

        meme_filename = pt.join(opts.out_dir, 'meme.txt')
        meme_file = open_meme(meme_filename, background_freq)

        for f in range(pwms.shape[-1]):
            pwms[:,:,f] = pwms[:,:,f] / pwms[:,:,f].sum(axis=1).reshape(-1, 1)

            if nb_sites[f] == 0:
                continue
            add_to_meme(meme_file, f, pwms[:,:,f], nb_sites[f],
                        trim_thr=opts.trim_thr)
        meme_file.close()


        log.info('Analyzing filters')
        log.info('-----------------')

        weblogo_opts = WEBLOGO_OPTS
        if self.opts.weblogo_opts:
            weblogo_opts = self.opts.weblogo_opts
        log.info('Plotting sequence logos')

        # logo_file = pt.join(sub_dirs['fa'], '%03d.fa' % f)
        # write_kmers(kmers[f], logo_file)
        for f in range(pwms.shape[-1]):
            logo_file = pt.join(sub_dirs['fa'], '%03d.fa' % f)
            plot_logo(logo_file,
                      self.plot_filename(sub_dirs['logos'], '%03d' % f),
                      options=weblogo_opts)
            if self.opts.delete_fasta:
                os.remove(logo_file)

        filter_stats = []
        for f in range(pwms.shape[-1]):
            # filter_act = filters_act[:nb_sample, :, f]
            if nb_sites[f] == 0:
                continue
            stats = OrderedDict()
            stats['idx'] = f
            filter_weights = filters_weights[:, :, f].T
            stats['motif'] = get_motif_from_weights(filter_weights)

            stats['act_mean'] = 0  #0 for simplification
            stats['act_std'] = 0  # 0 for simplification
            stats['ic'] = info_content(pwms[:, :, f])
            stats['nb_site'] = nb_sites[f]

            stats = pd.Series(stats)
            filter_stats.append(stats)


            # if stats['act_mean'] == 0:
            #     log.info('Dead filter -> skip')
            #     continue

            #


            # if self.opts.plot_dens:
            #     log.info('Plotting filter densities')
            #     tmp = self.plot_filename(sub_dirs['dens'], '%03d' % f)
            #     plot_filter_densities(np.ravel(filter_act[f]), tmp)  # Double check if correctly implements

            if self.opts.plot_heat:
                log.info('Plotting filter heatmap')
                tmp = self.plot_filename(sub_dirs['heat'], '%03d' % f)
                plot_filter_heatmap(filter_weights, tmp)
        
        filter_stats = pd.DataFrame(filter_stats)
        for name in ['idx', 'nb_site']:
            filter_stats[name] = filter_stats[name].astype(np.int32)
        filter_stats.sort_values('act_mean', ascending=False, inplace=True)
        print()
        print('\nFilter statistics:')
        print(filter_stats.to_string())
        filter_stats.to_csv(pt.join(opts.out_dir, 'stats.tsv'),
                            float_format='%.4f',
                            sep='\t', index=False)



        # if self.opts.plot_pca:
        #     tmp = min(len(filters_act), self.opts.nb_sample_pca)
        #     log.info('Performing PCA on activations using %d samples' % tmp)
        #     # Down-sample activations to at most nb_sample_pca samples to reduce
        #     # memory usage and run-time.
        #     pca_act = filters_act[:tmp, :, list(filters_idx)]
        #
        #     act_mean = pca_act.mean(axis=1)
        #     tmp = self.plot_filename(self.opts.out_dir, 'pca_mean')
        #     plot_pca(act_mean, labels=filters_idx, filename=tmp)
        #
        #     weights = linear_weights(pca_act.shape[1])
        #     act_av = np.average(pca_act, 1, weights)
        #     tmp = self.plot_filename(self.opts.out_dir, 'pca_wmean')
        #     plot_pca(act_av, labels=filters_idx, filename=tmp)
        #
        #     act_max = pca_act.max(axis=1)
        #     tmp = self.plot_filename(self.opts.out_dir, 'pca_max')
        #     plot_pca(act_max, labels=filters_idx, filename=tmp)


        if opts.motif_dbs:
            log.info('Running tomtom')
            cmd = 'tomtom -dist pearson -thresh {thr} -oc {out_dir} ' + \
                '{meme_file} {motif_dbs}'
            # cmd = 'srun /shared/apps/auto/singularity/3.8.4-gcc-7.3.0-fick/bin/singularity run /cm/shared/containers/meme/5.5.0/crun.meme tomtom -dist pearson -thresh {thr} -oc {out_dir} ' + \
            #       '{meme_file} {motif_dbs}'
            cmd = cmd.format(thr=opts.fdr,
                             out_dir=pt.join(opts.out_dir, 'tomtom'),
                             meme_file=meme_filename,
                             motif_dbs=' '.join(opts.motif_dbs))
            print('\n', cmd)
            subprocess.call(cmd, shell=True)

            meme_motifs = []
            
            # if opts.compare_models:
            #     for motif_db in opts.motif_dbs:
            #         meme_motifs.append(read_meme_from_other_model(motif_db))
            # else:
            #     for motif_db in opts.motif_dbs:
            #         meme_motifs.append(read_meme_db(motif_db))
            
            for motif_db in opts.motif_dbs:
                meme_motifs.append(read_meme_db(motif_db))
                    
            meme_motifs = pd.concat(meme_motifs)
            tmp = pt.join(opts.out_dir, 'tomtom', 'meme_motifs.tsv')
            meme_motifs.to_csv(tmp, sep='\t', index=False)

            report = get_report(
                pt.join(opts.out_dir, 'stats.tsv'),
                pt.join(opts.out_dir, 'tomtom', 'tomtom.tsv'),
                meme_motifs)

                
            report.sort_values(['idx', 'q-value', 'act_mean'],
                               ascending=[True, True, False], inplace=True)
            report.to_csv(pt.join(opts.out_dir, 'report.tsv'), index=False,
                          sep='\t', float_format='%.3f')

            report_top = report.groupby('idx').first().reset_index()
            report_top.sort_values(['q-value', 'act_mean'],
                                   ascending=[True, False], inplace=True)
            report_top.index = range(len(report_top))
            report_top.to_csv(pt.join(opts.out_dir,
                                      'report_top.tsv'), index=False,
                              sep='\t', float_format='%.3f')

            print('\nTomtom results:')
            print(report_top.to_string())



if __name__ == '__main__':
    app = App()
    app.run(sys.argv)


            

            



            