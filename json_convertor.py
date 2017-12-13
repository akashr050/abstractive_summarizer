
# coding: utf-8

# In[6]:


import json
import numpy as np
import sys


# In[60]:


with open(sys.argv[1]) as json_data:
    d = json.load(json_data)

j = 0
i = 0
max_i = len(d['data'])
lines = []
with open(sys.argv[2],"r") as f:
    for line in f:
        x = line.replace('\n','')
        lines.append(x)
        


for line_n in range(len(lines)):
    if i < (max_i):
        max_j = len(d['data'][i]['paragraphs'])
        if j < (max_j):
            d['data'][i]['paragraphs'][j]['context'] = lines[line_n]
            j +=1
        if j >= max_j:
            i+= 1



# In[61]:


with open('data_mod.json', 'w',encoding='utf8') as outfile:  
    json.dump(d, outfile)


# In[57]:


#print(d['data'][0]['title'])
#print(len(d['data'][0]['paragraphs']))
#print(len(d['data']))
#for j in range(48):
#    for i in range(54):
#        print(d['data'][0]['paragraphs'][i]['context'])

