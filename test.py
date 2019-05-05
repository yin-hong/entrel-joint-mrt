import json
import numpy as np

def replace_latin(string):
    r = {'á': 'a',
         'â': 'a',
         'Á': 'A',
         'É': 'E',
         'è': 'e',
         'é': 'e',
         'ê': 'e',
         'í': 'i',
         'ó': 'o',
         'ô': 'o',
         'ö': 'o',
         'Ó': 'O',
         'ú': 'u',
         'ü': 'u',
         'ñ': 'n'}
    for k, v in r.items():
        string = string.replace(k, v)
    return string


# with open('data/nyt/test.json', 'r', encoding='utf8', errors='ignore') as f:
#     lines = f.readlines()
#     num = 0
#     sent_num = 0
#     print('lines num: ', len(lines))
#     for i, line in enumerate(lines):
#         sent = json.loads(line.rstrip('\r\n'))
#         entityMentions = sent['entityMentions']
#         flag = False
#         for entity in entityMentions:
#             text = entity['text']
#             text = replace_latin(text)
#             if text.endswith('.'):
#                 num += 1
#                 print(i, text)
#                 flag = True
#         if flag:
#             sent_num += 1
#     print('num :', num)
#     print('sent_num:', sent_num)

#
emb = np.random.uniform(-1, 1, (10, 10))
print(emb)
np.save('data/temp.npy', emb)
# emb = np.load('data/temp.npy')
# print(emb.shape)