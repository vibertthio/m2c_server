import os
import time
from flask import Flask, request, Response
from flask_cors import CORS
import numpy as np
import json
import pypianoroll
from pypianoroll import Multitrack, Track
import torch
import torch.utils.data as Data

from BiLSTM_M2C.eval_midi import m2c_generator

MAX_NUM_SAMPLE = 7
app = Flask(__name__)
app.config['ENV'] = 'development'
CORS(app)

'''
load model
'''
m2c_gen = m2c_generator( MAX_NUM_SAMPLE )
m2c_gen.next()

'''
api route
'''
@app.route('/static/<s1>', methods=['GET'], endpoint='static_1')
def static(s1):

    response =m2c_gen.send( int(s1) )
    m2c_gen.next()
    #print response 
    response_pickled = json.dumps(response)
    print response_pickled
    return Response(response=response_pickled, status=200, mimetype="application/json")


#@app.route('/static/<s1>/<s2>', methods=['GET'], endpoint='static_twosong_1', defaults={'num': '4'})
#@app.route('/static/<s1>/<s2>/<num>', methods=['GET'], endpoint='static_twosong_1')
#def static_twosong(s1, s2, num):
#    with torch.no_grad():
#        global UNIT_LEN
#        global INTERP_NUM
#        global TOTAL_LEN
#        global path
#        INTERP_NUM = int(num)
#        TOTAL_LEN = (INTERP_NUM + 2) * 4
#        
#        song1 = path + songfiles[int(s1)]
#        song2 = path + songfiles[int(s2)]
#        print('song1:',song1)
#        print('song2:',song2)
#        m, c = model.load_midi(song1, song2, UNIT_LEN)
#        m_roll, c_roll = model.interp_sample(vae, s1, s2, m, c, INTERP_NUM)
# 
#    response_pickled = numpy2json(m_roll, c_roll, TOTAL_LEN)
#    return Response(response=response_pickled, status=200, mimetype="application/json")
'''
start app
'''
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003)