import os
import numpy as np

CONFIG_ALL = dict()


#exp_name  = 'BiLSTM_Harmonizer_no_residual' 
exp_name  = 'BiLSTM_Harmonizer'

func_alpha      = 2.0

# Change these for testing 
test_dir  = '/home/vibertthio/local_dir/vibertthio/m2c/v1/BiLSTM_M2C/data/'
chord_dic = '/home/vibertthio/local_dir/vibertthio/m2c/v1/BiLSTM_M2C/chord_dic.pkl'
log_dir   = '/home/vibertthio/local_dir/vibertthio/m2c/v1/BiLSTM_M2C/log/'

train_file_path = '/home/yyeh/lead_sheet/theorytab/harm_tt_train.npy'
test_file_path  = '/home/yyeh/lead_sheet/theorytab/harm_tt_test.npy'
dataset_dir     = '/home/yyeh/lead_sheet/theorytab/leadsheet_challenge_tt/'
eval_model      = 'model_110.ckpt'
result_dir      = 'result/' + exp_name
chord_resolution= 2

num_total_class = 49
batch_size      = 16
num_workers     = 4
learning_rate   = 0.01

num_epochs      = 250
input_size      = 12
lstm_hidden_size= 64
fc_hidden_size  = 256
num_layers      = 2
num_classes_c   = 49
num_classes_cf  = 4 
# ------------------------------

CONFIG_ALL['data'] = {
    'train_file_path' : train_file_path,
    'test_file_path'  : test_file_path,
    'batch_size'      : batch_size,
    'num_workers'     : num_workers,
    'num_total_class_cf' : num_classes_cf,
    'num_total_class_c'  : num_classes_c,
    'chord_dic'       : chord_dic,
    'chord_resolution': chord_resolution,
    'result_dir'      : result_dir,
    'dataset_dir'     : dataset_dir,
    'test_dir'        : test_dir
}

CONFIG_ALL['model'] = {
    'num_epochs' : num_epochs,
    'log_dir'  : log_dir,
    'exp_name' : exp_name,

    'input_size'  : input_size,
    'lstm_hidden_size' : lstm_hidden_size,
    'fc_hidden_size'   : fc_hidden_size, 
    'num_layers'       : num_layers,
    'num_classes_cf'   : num_classes_cf,
    'num_classes_c'    : num_classes_c,
    'learning_rate'    : learning_rate,
    'eval_model'       : eval_model, 
    'func_alpha'       : func_alpha
}

#for k, v in CONFIG_ALL['data'].items():
#    CONFIG_ALL['model'][k] = v
#
#for key, path in CONFIG_ALL['model']['dirs'].items():
#    path = os.path.join(CONFIG_ALL['model']['exp_name'], path)
#    CONFIG_ALL['model']['dirs'][key] = path
