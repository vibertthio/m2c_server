import requests
import numpy as np
import json

'''
show drum roll in console
'''
#def printleadsheet(roll):
#    trans = np.flip(np.transpose(roll), 0)
#    for r_i, r in enumerate(trans):
#        print('[{}]'.format(127 - r_i), end='') 
#        for i, w in enumerate(r):
#            if i > 0 and i % 12 == 0:
#                print('|', end='')
#            if w == 0:
#                print('_', end='')
#            else:
#                print('*', end='')
#        print()

addr = 'http://localhost:5003'
SONG_IDX = '5'

# test_url = addr + '/static'
test_url = addr + '/static' + '/' + SONG_IDX
content_type = 'application/json'
headers = {'content-type': content_type}

response = requests.get(
    test_url,
    headers=headers)

print(response)
r_json = json.loads(response.text)
m = r_json['melody']
c = r_json['chord']
print (len(m), len(m[0]))
print m
print c

#for i, d in enumerate(melody_rolls):
#    x = i
#    y = 0
#    print('({}, {})'.format(x, y))
#    printleadsheet(d)
#
#print('length of chords: {}'.format(len(chord_rolls)))
#for i, d in enumerate(chord_rolls):
#    x = i
#    y = 0
#    print('({}, {})'.format(x, y))
#    printleadsheet(d)
