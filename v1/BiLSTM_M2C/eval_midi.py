from config import CONFIG_ALL

import os 
import numpy as np 
import torch
import pandas as pd 
import pypianoroll 

from tqdm import tqdm 
import pretty_midi
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn

from model import LeadSheetHarmDataset, BiRNN, collate_fn_midi
from util import ensure_dir, pro_chordlabel_to_midi, write_midi, find_files, midi_feature
import datetime
import time
import logging

REST_IDX = 48 
REST_MEL = 0

def midi_to_list( pm, chord_list ) :
    MELODY_BEAT_RESOLUTION = 24
    MELODY_BAR_RESOL = 4*MELODY_BEAT_RESOLUTION
    CHORD_BAR_RESOL  = 2

    # Parse Melody 
    pno_roll = pypianoroll.Multitrack(beat_resolution=MELODY_BEAT_RESOLUTION)
    pno_roll.parse_pretty_midi(pm) 
    out_mel_list = list( np.argmax( pno_roll.tracks[0].pianoroll, axis=1 ) )

    # Pad to same length     
    max_len = int( max( np.ceil( float( len(out_mel_list) )/MELODY_BAR_RESOL ), np.ceil( float( len(chord_list) )/CHORD_BAR_RESOL ) ) )

    for i in range( len(chord_list), max_len*CHORD_BAR_RESOL) :
        chord_list.append(REST_IDX)

    for i in range( len(out_mel_list), max_len*MELODY_BAR_RESOL) :  
        out_mel_list.append( REST_MEL )

    #chord_list   = np.reshape( chord_list  , (max_len, CHORD_BAR_RESOL ) ) 
    #out_mel_list = np.reshape( out_mel_list, (max_len, MELODY_BAR_RESOL) )
    chord_list_out = [ [   chord_list[ i_bar*CHORD_BAR_RESOL  + i_chord ] for i_chord in range(0, CHORD_BAR_RESOL)  ] for i_bar in range(0, max_len) ]
    mel_list_out   = [ [ out_mel_list[ i_bar*MELODY_BAR_RESOL + i_mel   ] for i_mel   in range(0, MELODY_BAR_RESOL) ] for i_bar in range(0, max_len) ]
    return { 'melody' : mel_list_out, 'chord' : chord_list_out }

def m2c_generator( max_num_sample ):
    '''
        m2c Generator 
        Input  : a testing sample index 
        Output : Chord Label (n, 16)
                 Monophony Melody Label (n, 2)
                 BPM float 
        Average Elasped Time for one sample : 0.16 sec 
    '''
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')

    # Load Data
    chord_dic = pd.read_pickle( CONFIG_ALL['data']['chord_dic'] )

    # prepare features 
    all_files = find_files( CONFIG_ALL['data']['test_dir'], '*.mid'  )
    input_dic = []
    for i_file in all_files :
        _ = midi_feature(i_file, sampling_fac=2)
        _ = np.reshape( _, (1, _.shape[0], _.shape[1]) )
        input_dic.append( { 'midi' : i_file, 'm_embed' : _ } )
    print 'Total Number of files : ', len( input_dic )

    # training 
    model = BiRNN(CONFIG_ALL['model']['input_size'], 
                  CONFIG_ALL['model']['lstm_hidden_size'], 
                  CONFIG_ALL['model']['fc_hidden_size'], 
                  CONFIG_ALL['model']['num_layers'], 
                  CONFIG_ALL['model']['num_classes_cf'],
                  CONFIG_ALL['model']['num_classes_c'], 
                  device).to(device)
    
    # Load Model 
    path = os.path.join( CONFIG_ALL['model']['log_dir'], CONFIG_ALL['model']['exp_name'], 'models/', CONFIG_ALL['model']['eval_model'] )
    model.load_state_dict(torch.load(path))

    # Test the model
    with torch.no_grad(): 
        while True :
            test_idx = yield 

            if test_idx >= max_num_sample or test_idx < 0:
                print "Invalid sample index"
                continue
            m_embedding = input_dic[test_idx]['m_embed']
            out_cf, out_c = model( torch.tensor( m_embedding, dtype=torch.float).to(device)  )

            out_c = out_c.data.cpu().numpy()
        
            _, pred_cf  = torch.max( out_cf.data, 1 )
            pred_cf = pred_cf.data.cpu().numpy()

            i_out_tn1 = -1  
            i_out_tn2 = -1
            i_out_tn3 = -1
            i_out_t   = -1

            predicted = []
            c_threshold = 0.825
            f_threshold = 0.35
            #ochord_threshold = 1.0

            for idx, i_out in enumerate( out_c ) :
                # Seventh chord
                #T_chord_label = [0, 1, 2, 3, 4, 5, 102, 103, 104]
                #D_chord_label = [77, 78, 79, 55, 56, 57]
                #R_chord_label = [132]

                # Triad Chord 
                T_chord_label = [0, 1, 37]
                D_chord_label = [20, 28]
                R_chord_label = [48]

                O_chord_label = [ i for i in range(0, 48) if not (i in T_chord_label) or (i in D_chord_label) or(i in R_chord_label) ]

                # Bean Search for repeated note 
                if   pred_cf[idx] == 0 :
                    L = np.argsort( -np.asarray( [ i_out[i] for i in T_chord_label] ) )
                    if i_out_tn1 == T_chord_label[ L[0] ] and i_out_tn2 == T_chord_label[ L[0] ]:
                        i_out_t = T_chord_label[ L[1] ]  
                    else :
                        i_out_t = T_chord_label[ L[0] ] 

                elif pred_cf[idx] == 1 :
                    i_out_t = D_chord_label[ np.argmax( [ i_out[i] for i in D_chord_label] ) ] 

                elif pred_cf[idx] == 3 :
                    L = np.argsort( -np.asarray( [ i_out[i] for i in O_chord_label] ) )
                    if i_out_tn1 == O_chord_label[ L[0] ] and i_out_tn2 == O_chord_label[ L[0] ] :
                        i_out_t = O_chord_label[ L[1] ]  
                    else :
                        i_out_t = O_chord_label[ L[0] ] 

                else :
                    i_out_t = 48 

                predicted.append( i_out_t )
                i_out_tn2 = i_out_tn1
                i_out_tn1 = i_out_t
                i_out_last = i_out
            
            # Write file to midi 
            midi_original = pretty_midi.PrettyMIDI(  input_dic[test_idx]['midi']  )
            midi_chord    = pro_chordlabel_to_midi( predicted, chord_dic, inv_beat_resolution = CONFIG_ALL['data']['chord_resolution'], constant_tempo=midi_original.get_tempo_changes()[1])
            midi_chord.instruments[0].name = "Predicted_w_func"
            midi_original.instruments.append( midi_chord.instruments[0] )

            out_path = os.path.join( 'eval_test/', str(test_idx) + '.mid' )
            ensure_dir(out_path)
            midi_original.write( out_path )
            print "Write Files to : ", out_path
            
            out_mc = midi_to_list( midi_original, predicted )

            yield { 'melody' : out_mc['melody'],
                    'chord'  : out_mc['chord' ],
                    'BPM'    : float( midi_original.get_tempo_changes()[1] ) }

#def main():
#    m2c_gen = m2c_generator( 5 )
#    m2c_gen.next()
#    elasped_time_list = []
#    for i in range(0, 200):
#        start_time = time.time()
#        m2c_gen.send(i)
#        m2c_gen.next()
#        end_time   = time.time()
#        elasped_time = end_time - start_time
#        elasped_time_list.append(elasped_time) 
#    print np.mean(elasped_time_list)
#
#if __name__ == "__main__":
#    main()
