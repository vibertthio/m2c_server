import os, fnmatch
import numpy as np 
import pandas as pd
import pypianoroll
import pretty_midi
from collections import OrderedDict

def ensure_dir(file_path):
    ''' Check if the directory is exist. If not, creat one. '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1
    return 0

def pro_midi_to_event_monophony(midi_filename) :
    midi_data = pretty_midi.PrettyMIDI(midi_filename)
    tempo = midi_data.get_tempo_changes()[1]

    data = { 'tracks' : { "melody":[] } }
    for i_note in midi_data.instruments[0].notes : 
        data['tracks']['melody'].append(
            OrderedDict([
            #  basic info
            ('pitch', i_note.pitch%12),

            # event info
            ('isRest', False),
            ('event_on', int( i_note.start * tempo/60 ) ),
            ('event_off', int(i_note.end * tempo/60) ),
            ('event_duration', int(i_note.end * tempo/60 - i_note.start * tempo/60)  )
            ])
        )
    return data

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def midi_feature(midi_filename, sampling_fac=2):
    
    pnrl = pypianoroll.Multitrack(filepath=midi_filename)
    beat_resolution = 24*sampling_fac
    
    m_chroma = np.zeros( (pnrl.tracks[0].pianoroll.shape[0], 12 ) ) 
    for i in range(0, pnrl.tracks[0].pianoroll.shape[0]) :
        m_chroma[ i, int(np.argmax(pnrl.tracks[0].pianoroll[i, :]) % 12 )] = True
        
    melody_embedding = []

    for i_mm in range(0,int( pnrl.tracks[0].pianoroll.shape[0]/beat_resolution) ):
        embedding = np.mean( m_chroma[ i_mm*beat_resolution : (i_mm+1)*beat_resolution ], axis=0 )
        melody_embedding.append( np.reshape( embedding, (1, 12 ) ) ) 
    melody_embedding = np.vstack(melody_embedding)
    return melody_embedding


# Chord Dictionary Util #
def symbol_to_triad(s):
    s = s.split()[0]
    s = ''.join(i for i in s if not i.isdigit())
    if s[-3:] == 'maj':
        s = s[0:-3]
    if s[-1] == u'\xf8' :
        s = s[0:-1] + 'o'
    return s.lower()

def to_chroma(midi_code_step):
    return [ i%12 for i in midi_code_step ]
    

def note_symbol_to_chroma(s):
    if   s.lower() == 'c' :
        return 0 
    elif s.lower() =='db' :
        return 1
    elif s.lower() == 'd' :
        return 2
    elif s.lower() =='eb' :
        return 3
    elif s.lower() == 'e' :
        return 4
    elif s.lower() == 'f' :
        return 5
    elif s.lower() == 'gb':
        return 6
    elif s.lower() == 'g' :
        return 7
    elif s.lower() == 'ab':
        return 8 
    elif s.lower() == 'a' :
        return 9  
    elif s.lower() == 'bb':
        return 10 
    elif s.lower() == 'b' :
        return 11
    else :
        return None

def get_key_offset(key):
    return NOTE_TO_OFFSET[key]

def symbol_to_feature(s):
    s = symbol_to_triad(s)
    OCT_SHIFT = 36
    NUM_CHORD = 3
    
    MAG = 1.0
    ROOT_WEIGHT  = 1.0
    THIRD_WEIGHT = 0.75
    FIFTH_WEIGHT = 0.5
    
    chroma_feature = np.zeros(12)
    
    if s[-1]   =='m': # Minor Chord
        root = note_symbol_to_chroma(s[0:-1])
        midi_chord     = np.array( [0, 7, 15] ) + root + OCT_SHIFT
        
        chroma_feature[ midi_chord[0]%12 ] = MAG  * ROOT_WEIGHT
        chroma_feature[ midi_chord[1]%12 ] = MAG  * THIRD_WEIGHT
        chroma_feature[ midi_chord[2]%12 ] = MAG  * FIFTH_WEIGHT
        
        chord_idx      = root*NUM_CHORD 
        
    elif s[-1] == 'o': # Diminish Chord
        root = note_symbol_to_chroma(s[0:-1])
        midi_chord     = np.array( [0, 6, 15] ) + root + OCT_SHIFT
        
        chroma_feature[ midi_chord[0]%12 ] = MAG  * ROOT_WEIGHT
        chroma_feature[ midi_chord[1]%12 ] = MAG  * THIRD_WEIGHT
        chroma_feature[ midi_chord[2]%12 ] = MAG  * FIFTH_WEIGHT
        
        chord_idx      = root*NUM_CHORD + 1
        
    else :             # Major Chord 
        root = note_symbol_to_chroma(s)
        midi_chord     = np.array( [0, 7, 16] ) + root + OCT_SHIFT
        
        chroma_feature[ midi_chord[0]%12 ] = MAG  * ROOT_WEIGHT
        chroma_feature[ midi_chord[1]%12 ] = MAG  * THIRD_WEIGHT
        chroma_feature[ midi_chord[2]%12 ] = MAG  * FIFTH_WEIGHT
        
        chord_idx      = root*NUM_CHORD + 2
        
    return s, chord_idx, chroma_feature, midi_chord

# Proc Util # 
def wrapping_melody(melody_events, beats_sec):
    octave_melody = 6
    
    melody_track = pretty_midi.Instrument(program=0)
    init_note = octave_melody * 12

    for note in melody_events:
        if note is None:
            continue
        note_number = note['pitch'] + init_note

        # event_on/off
        start = note['event_on'] * beats_sec
        end = note['event_off'] * beats_sec

        note = pretty_midi.Note(velocity=100, pitch=int(note_number), start=start, end=end)
        melody_track.notes.append(note)
    return melody_track


def wrapping_chord(chord_events, chord_dic, beats_sec):
    
    chord_track = pretty_midi.Instrument(program=0)
    
    for chord in chord_events:
        if chord is None:
            continue
        
        comp = np.asarray( chord_dic[symbol_to_triad( chord['symbol'] ) == chord_dic['Symbol'] ]['Midi'] )[0]

        # event_on/off
        start = chord['event_on']  * beats_sec
        end   = chord['event_off'] * beats_sec

        for note_midi in comp:
            note = pretty_midi.Note(velocity=100, pitch=int(note_midi), start=start, end=end)
            chord_track.notes.append(note)

    return chord_track

def write_midi(lead_sheet, save_path='./', name='test') :
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, name+'.mid')
    lead_sheet.write(filename)
    print "Write to filename ", filename 
    
def proc_to_midi(
        melody_events,
        chord_events,
        chord_dic,
        key='C',
        to_chroma=False,
        bpm=120,
        beats_in_measure=4 ):

    bpm = float(bpm)
    if bpm == 0.0:
        bpm = 120

    beats_in_measure = int(beats_in_measure)
    beats_sec = 60.0 / bpm
    
    lead_sheet = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    
    chord_track  = wrapping_chord(chord_events, chord_dic, beats_sec)
    melody_track = wrapping_melody(melody_events, beats_sec)
    ts = pretty_midi.TimeSignature(beats_in_measure, 4, 0)
    ks = pretty_midi.KeySignature(get_key_offset(key), 0)
    
    lead_sheet.time_signature_changes.append(ts)
    lead_sheet.key_signature_changes.append(ks)

    lead_sheet.instruments.append(melody_track)
    lead_sheet.instruments.append(chord_track)
        
    return lead_sheet

def pro_chord_to_feature( event, chord_dict, beat_resolution=2) :
    BEAT_RESOLUTION = 48 
    REST_IDX = 36
    BEAT_IN_MM = 4
    MAX_BEAT = np.ceil(max( [ float(i['event_off']) for i in event['tracks']['chord'] if i != None ] )/BEAT_IN_MM)*BEAT_IN_MM
    
    # Processing Chord Feature 
   
    chord_map = np.zeros( (int(MAX_BEAT*BEAT_RESOLUTION), len(chord_dict)) )
    for i_chord in event['tracks']['chord'] :
        if i_chord == None:
            continue 
        idx = chord_dict[ chord_dict['Symbol'] == symbol_to_triad(i_chord['symbol']) ]['Label'].values[0],
        start_idx = int( i_chord['event_on']  * BEAT_RESOLUTION )
        end_idx   = int( i_chord['event_off'] * BEAT_RESOLUTION )
        
        chord_map[start_idx:end_idx+1, idx] = True 
    
    # Truncation 
    chord_embedding = []
    chord_label     = []

    for i_mm in range(0,int( MAX_BEAT/beat_resolution) ) :
        interval_sum = np.sum( chord_map[ i_mm*beat_resolution*BEAT_RESOLUTION : (i_mm+1)*beat_resolution*BEAT_RESOLUTION ], axis=0 )
        interval_sum_all = np.sum(interval_sum)
        if interval_sum_all == 0 :
            label = REST_IDX
            embedding = chord_dict[ chord_dict['Label'] == REST_IDX ]['Embedding'].values[0]
        else :
            label = np.argmax( interval_sum )
            embedding = chord_dict[ chord_dict['Label'] == label ]['Embedding'].values[0]
        
        chord_label.append(label)
        chord_embedding.append( np.reshape( embedding, (1, len(embedding) ) ) ) 
        
    chord_embedding = np.vstack(chord_embedding)
    
    # Processing Melody 
    melody_map = np.zeros( (int(BEAT_RESOLUTION*MAX_BEAT), 12) )
    
    for i_melody in event['tracks']['melody'] :
        if i_melody == None:
            continue 
        start_idx = int( i_melody['event_on']  * BEAT_RESOLUTION )
        end_idx   = int( i_melody['event_off'] * BEAT_RESOLUTION )
        melody_map[start_idx:end_idx+1, int(i_melody['pitch']%12)] += 1 
    
    # Truncation 
    melody_embedding = []
    for i_mm in range(0,int( MAX_BEAT/beat_resolution) ):
        embedding = np.mean( melody_map[ i_mm*beat_resolution*BEAT_RESOLUTION : (i_mm+1)*beat_resolution*BEAT_RESOLUTION ], axis=0 )
        melody_embedding.append( np.reshape( embedding, (1, 12 ) ) ) 
    melody_embedding = np.vstack(melody_embedding)
    
    return melody_embedding, chord_embedding, chord_label

def pro_chordlabel_to_midi( chord_label, chord_dict, inv_beat_resolution = 2, constant_tempo=120 ):
    
    VOL = 90

    beat_reol_pnrl = 48
    step_width = beat_reol_pnrl * inv_beat_resolution
    chord_pnrl = np.zeros( (len(chord_label)*beat_reol_pnrl*inv_beat_resolution, 128) )

    for idx_label, i_label in enumerate( chord_label ) :
        midi_chord = chord_dict[ chord_dict["Label"] == i_label ]['Midi'].values[0]

        if (idx_label % 2) == 0 and ( idx_label+1 < len(chord_label) ) :
            if ( chord_label[idx_label] == chord_label[idx_label+1] ) : 
                step_end = step_width
            else :
                step_end = step_width-1
        else :
            step_end = step_width-1
            
        for i_midi_chord in midi_chord :
            for i in range(0, step_end):
                chord_pnrl[int( idx_label*step_width + i )][int(i_midi_chord)] = VOL
    
    mt = pypianoroll.Multitrack(tracks=[pypianoroll.Track(pianoroll=chord_pnrl)], beat_resolution=beat_reol_pnrl)
    return mt.to_pretty_midi(constant_tempo=constant_tempo, constant_velocity=60 )