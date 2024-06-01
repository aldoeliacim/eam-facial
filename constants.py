# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import os
import re
import sys

import numpy as np

# Directory where all results are stored.
data_path = 'data'
run_path = 'runs'
idx_digits = 3
prep_data_fname = 'prep_data.npy'
pred_noised_data_fname = 'prep_noised_data.npy'
prep_labels_fname = 'prep_labels.npy'

image_path = 'images'
testing_path = 'test'
memories_path = 'memories'
dreams_path = 'dreams'

data_prefix = 'data'
labels_prefix = 'labels'
features_prefix = 'features'
memories_prefix = 'memories'
noised_prefix = 'mem_noised'
mem_conf_prefix = 'mem_confrix'
model_prefix = 'model'
recognition_prefix = 'recognition'
recog_noised_prefix = 'recog_noised'
weights_prefix = 'weights'
weights_noised_prefix = 'weights-noised'
classification_prefix = 'classification'
classification_noised_prefix = 'classif-noised'
stats_prefix = 'model_stats'
learn_params_prefix ='learn_params'
memory_parameters_prefix='mem_params'
chosen_prefix = 'chosen'

balanced_data = 'balanced'
seed_data = 'seed'
learning_data_seed = 'seed_balanced'
learning_data_learned = 'learned'

# Categories suffixes.
original_suffix = '-original'
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'
noised_suffix = '-noised'
prod_noised_suffix = '-prod_noised'
memories_suffix = '-memories'

# Model suffixes.
encoder_suffix = '-encoder'
classifier_suffix = '-classifier'
decoder_suffix = '-decoder'
memory_suffix = '-memory'

data_suffix = '_X'
labels_suffix = '_Y'
matrix_suffix = '-confrix'

agreed_suffix = '-agr'
original_suffix = '-ori'
amsystem_suffix = '-ams'
nnetwork_suffix = '-rnn'
learning_suffixes = [[original_suffix], [agreed_suffix], [amsystem_suffix],
    [nnetwork_suffix], [original_suffix, amsystem_suffix]]

# Number of columns in memory
domain = 256
n_folds = 1
n_jobs = 1
dreaming_cycles = 6

iota_default = 0.0
kappa_default = 0.0
xi_default = 0.0
sigma_default = 0.25
params_defaults = [iota_default, kappa_default, xi_default, sigma_default]
iota_idx = 0
kappa_idx = 1
xi_idx = 2
sigma_idx = 3

nn_training_percent = 0.70
am_filling_percent = 0.20
am_testing_percent = 0.10
noise_percent = 50

n_labels = 2
labels_per_memory = 1
all_labels = list(range(n_labels))
label_formats = ['r:v', 'y--d', 'g-.4', 'y-.3', 'k-.8', 'y--^',
    'c-..', 'm:*', 'c-1', 'b-p', 'm-.D', 'c:D', 'r--s', 'g:d',
    'm:+', 'y-._', 'm:_', 'y--h', 'g--*', 'm:_', 'g-_', 'm:d']

precision_idx = 0
recall_idx = 1
accuracy_idx = 2
entropy_idx = 3
no_response_idx = 4
no_correct_response_idx = 5
correct_response_idx = 6
n_behaviours = 7

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
sigma_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
n_best_memory_sizes = 3
n_samples = 10
learned_data_groups = 6

class ExperimentSettings:
    def __init__(self, params = None):
        if params is None:
            print('Memory parameters not provided, ' 
                + 'so defaults are used for all memories.')
            self.mem_params = params_defaults
        else:
            # If not None, it must be a one dimensional array.
            assert(isinstance(params,np.ndarray))
            assert(params.ndim == 1)
            # The dimension must have four elements
            # iota, kappa, xi, sigma
            shape = params.shape
            assert(shape[0] == 4)
            self.mem_params = params

    def __str__(self):
        s = '{Parameters: ' + str(self.mem_params) + '}'
        return s

def print_warning(*s): print('WARNING:', *s, file = sys.stderr)

def print_error(*s): print('ERROR:', *s, file = sys.stderr)

def print_counter(n, every, step = 1, symbol = '.', prefix = ''):
    if n == 0:
        return
    e = n % every
    s = n % step
    if (e != 0) and (s != 0):
        return
    counter = symbol
    if e == 0:
        counter =  ' ' + prefix + str(n) + ' '
    print(counter, end = '', flush=True)

def int_suffix(n, prefix=None):
    prefix = '' if prefix is None else '-' + prefix + '_'
    return prefix + str(n).zfill(3)

def float_suffix(x, prefix=None):
    prefix = '' if prefix is None else '-' + prefix + '_'
    return prefix + f'{x:.2f}'

def extended_suffix(extended): return '-ext' if extended else ''

def numeric_suffix(prefix, value): return '-' + prefix + '_' + str(value).zfill(3)

def fold_suffix(fold):
    return '' if fold is None else int_suffix(fold, 'fld')

def learned_suffix(learned): return int_suffix(learned, 'lrn')

def stage_suffix(stage): return int_suffix(stage, 'stg')

def msize_suffix(msize): return int_suffix(msize, 'msz')

def sigma_suffix(sigma): return float_suffix(sigma, 'sgm')

def learned_suffix(learned): return numeric_suffix('lrn', learned)

def stage_suffix(stage): return numeric_suffix('stg', stage)

def dream_depth_suffix(cycle): return numeric_suffix('dph', cycle)

def get_name_w_suffix(prefix):
    suffix = ''
    return prefix + suffix


def get_full_name(prefix, es):
    if es is None:
        return prefix
    name = get_name_w_suffix(prefix)
    return name

# Currently, names include nothing about experiment settings.
def model_name(es): return model_prefix
def stats_model_name(es): return stats_prefix
def data_name(es): return data_prefix
def features_name(es): return features_prefix
def labels_name(es): return labels_prefix
def memories_name(es): return memories_prefix
def noised_memories_name(es): return noised_prefix
def recognition_name(es): return recognition_prefix
def noised_recog_name(es): return recog_noised_prefix
def weights_name(es): return weights_prefix
def noised_weights_name(es): return weights_noised_prefix
def classification_name(es): return classification_prefix
def noised_classification_name(es): return classification_noised_prefix
def learn_params_name(es): return learn_params_prefix
def mem_params_name(es): return memory_parameters_prefix

def dirname(path):
    match = re.search('[^/]*$', path) 
    if match is None:
        return path
    tuple = os.path.splitext(match.group(0))
    return os.path.dirname(path) if tuple[1] else path

def create_directory(path):
    try:
        os.makedirs(path)
        print(f'Directory {path} created.')
    except FileExistsError:
        print(f'Directory {path} already exists.')
    
def filename(name_prefix, es = None, fold = None, extension = ''):
    """ Returns a file name in run_path directory with a given extension and an index
    """
    # Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs(run_path)
        print("Directory " , run_path ,  " created ")
    except FileExistsError:
        pass
    return run_path + '/' + get_full_name(name_prefix,es) \
        + fold_suffix(fold) + extension

def csv_filename(name_prefix, es = None, fold = None):
    return filename(name_prefix, es, fold, '.csv')

def data_filename(name_prefix, es = None, fold = None):
    return filename(name_prefix, es, fold, '.npy')

def json_filename(name_prefix, es):
    return filename(name_prefix, es, extension='.json')

def pickle_filename(name_prefix, es = None, fold = None):
    return filename(name_prefix, es, fold, '.pkl')

def picture_filename(name_prefix, es, fold = None):
    return filename(name_prefix, es, fold, extension='.svg')

def image_filename(prefix, idx, label, suffix='', es=None, fold=None):
    img_dir = image_path + '/' + prefix + '/'
    full_path = run_path + '/' + img_dir
    try:
        print('Creating dir: ', full_path)
        os.makedirs(full_path, exist_ok=True)
    except Exception as e:
        print("Error creating dir: ", e)

    name_prefix = img_dir + str(label).zfill(3) + '_' + str(idx).zfill(5)  + suffix
    return filename(name_prefix, es, fold, extension='.png')
    
def learned_data_filename(suffix, es, fold):
    prefix = learning_data_learned + suffix + data_suffix
    return data_filename(prefix, es, fold)

def learned_labels_filename(suffix, es, fold):
    prefix = learning_data_learned + suffix + labels_suffix
    return data_filename(prefix, es, fold)

def seed_data_filename():
    return data_filename(learning_data_seed + data_suffix)

def seed_labels_filename():
    return data_filename(learning_data_seed + labels_suffix)

def model_filename(name_prefix, es, fold):
    return filename(name_prefix, es, fold)

def encoder_filename(name_prefix, es, fold):
    return filename(name_prefix + encoder_suffix, es, fold)

def classifier_filename(name_prefix, es, fold):
    return filename(name_prefix + classifier_suffix, es, fold)

def decoder_filename(name_prefix, es, fold):
    return filename(name_prefix + decoder_suffix, es, fold)

def memory_confrix_filename(fill, es, fold):
    prefix = mem_conf_prefix + int_suffix(fill, 'fll')
    return data_filename(prefix, es, fold)

def recog_filename(name_prefix, es, fold):
    return csv_filename(name_prefix, es, fold)

def testing_image_filename(path, idx, label, es, fold):
    return image_filename(path, idx, label, original_suffix, es, fold)

def prod_testing_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, testing_suffix, es, fold)

def noised_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, noised_suffix, es, fold)

def prod_noised_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, prod_noised_suffix, es, fold)

def memory_image_filename(dir, idx, label, es, fold):
    return image_filename(dir, idx, label, memory_suffix, es, fold)

def mean_idx(m): return m

def std_idx(m): return m+1

def padding_cropping(data, n_frames):
    frames, _  = data.shape
    df = frames - n_frames
    if df < 0:
        return []
    elif df == 0:
        return [data]
    else:
        features = []
        for i in range(df+1):
            features.append(data[i:i+n_frames,:])
        return features

def get_data_in_range(data, i, j):
    if j > i:
        return data[i:j]
    else:
        pre = data[i:]
        pos = data[:j]
        if len(pre) == 0:
            return pos
        elif len(pos) == 0:
            return pre
        else:
            return np.concatenate((pre,pos), axis=0)

def print_csv(data):
    writer = csv.writer(sys.stdout)
    if np.ndim(data) == 1:
        writer.writerow(data)
    else:
        writer.writerows(data)
