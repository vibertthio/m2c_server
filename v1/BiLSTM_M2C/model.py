import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LeadSheetHarmDataset(Dataset):

    def __init__(self, npy_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.load(npy_file).item()
        print "File loaded from ", npy_file

    def __len__(self):
        return len(self.data['m_embedding'])

    def __getitem__(self, idx):
        return { 'm_embedding' : self.data['m_embedding'][idx], 
                 'chord_label' : self.data['chord_label'][idx], 
                 'chord_func'  : self.data['chord_func'][idx], 
                 'midi'        : self.data['midi'][idx]  }     

def collate_fn_midi(batch):
    
    _m    = [ torch.tensor( i['m_embedding'], dtype=torch.float32 ).view( 1, i['m_embedding'].shape[0], i['m_embedding'].shape[1] ) for i in batch ]   
    _c    = [ i['chord_label'] for i in batch ]
    _midi = [ i['midi'] for i in batch ]
    
    return { 'x': _m, 'y': _c, 'midi' : _midi }   

def collate_fn(batch):
    
    _m    = [ torch.tensor( i['m_embedding'], dtype=torch.float32 ).view( 1, i['m_embedding'].shape[0], i['m_embedding'].shape[1] ) for i in batch ]   
    _c    = [ i['chord_label'] for i in batch ]
    _cf   = [ i['chord_func']  for i in batch ]
    return { 'x': _m, 'y_c': _c, 'y_cf':_cf }    


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, fc_hidden_size, num_layers, num_classes_cf, num_classes_c, device):

        super(BiRNN, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.fc_hidden_size   = fc_hidden_size
        self.num_layers       = num_layers

        self.only_use_last_hidden_state = True 
        self.batch_first                = True
        self.device                     = device
        self.num_classes_cf             = num_classes_cf
        self.num_classes_c              = num_classes_c

        self.lstm = nn.LSTM(input_size , lstm_hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        self.fc1_cf = nn.Linear(lstm_hidden_size*num_layers + input_size, fc_hidden_size).to(device)  # 2 for bidirection
        #self.fc1 = nn.Linear(lstm_hidden_size*num_layers, fc_hidden_size).to(device)  # 2 for bidirection
        self.fc2_cf = nn.Linear(fc_hidden_size, num_classes_cf  ).to(device)  # 2 for bidirection

        self.fc1_c = nn.Linear(lstm_hidden_size*num_layers + input_size + num_classes_cf, fc_hidden_size).to(device)  # 2 for bidirection
        #self.fc1 = nn.Linear(lstm_hidden_size*num_layers, fc_hidden_size).to(device)  # 2 for bidirection
        self.fc2_c = nn.Linear(fc_hidden_size, num_classes_c  ).to(device)  # 2 for bidirection

    def forward(self, x):
        """
            sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort
            :param x: Variable
            :param x_len: numpy list
            :return:
        """
        # Set initial states
        NUM_DIRECTION = 2
        h0 = torch.zeros( self.num_layers*NUM_DIRECTION, x.size(0), self.lstm_hidden_size ).to(self.device) # 2 for bidirection 
        c0 = torch.zeros( self.num_layers*NUM_DIRECTION, x.size(0), self.lstm_hidden_size).to(self.device)

        out_lstm, (ht, ct) = self.lstm(x, (h0, c0))
        out = out_lstm.view( out_lstm.size(1), out_lstm.size(2) )
        

        # Add Residual
        x       = x.view( x.size(1), x.size(2) )
        out_rnn = torch.cat((out, x), 1)

        relu1 = nn.ReLU()
        out_cf = relu1( self.fc1_cf(out_rnn) )
        out_cf = self.fc2_cf(out_cf)

        out_rnn_c = torch.cat((out_rnn, out_cf), 1)
        relu2 = nn.ReLU()
        out_c = relu2( self.fc1_c(out_rnn_c) )
        out_c = self.fc2_c(out_c)

        return out_cf, out_c