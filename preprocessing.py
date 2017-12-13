## This code helps read the Signal-Media 1 News Article Dataset and preprocess it into a form suitable for our task


#Run from terminal as : (Assuming you have the Signal Media Dataset in the same folder)
# python preprocessing.py signalmedia-1m.jsonl

# Libraries
import pickle
from tqdm import tqdm_notebook as tqdm
import re
import json_lines
import numpy as np
import sys

# Data Path
# data_path = './signalmedia-1m.jsonl'
data_path = sys.argv[1]

def main():
    # Reading the files
    text = []
    title = []
    with open(data_path, 'rb') as f:
        print("\n Opened the File!\t",data_path)
        for item in json_lines.reader(f):
            t = item['content'].replace('\n','')
            t = t.replace("\'",'"')
            t = t.replace("  ",'')
            text.append(t)
            title.append(item['title'])
    # Call the cleaning function

    print(" Data has been read!")
    new_txt,new_title= clean_texts_word_limit(text,title)
    print(" Data has been processed")
    # Save the abstracts and titles serially as pickle files
    pickle.dump(new_txt,open("AbsSumm_text.pkl",'wb'))
    pickle.dump(new_title,open("AbsSumm_title.pkl",'wb'))
    print(" Articles and Titles are ready!")
# Functions
# Functions returns the cleansed/preprocessed version of the abstract
def clean_texts_word_limit(txts,titls,verbose=False):
    # load the symbols and multi-lingual pickled list of characters
    remove = pickle.load(open('./remove_symbols.pkl','rb'))
    danger = pickle.load(open('./danger_symbols.pkl','rb'))
    new_txt = []
    new_titl = []
    range_len_txts = range(len(txts))
    if verbose:
        range_len_txts = tqdm(range_len_txts)
    for i in range_len_txts:
        no_bracket_t = re.sub(r'\(.*\)','', txts[i])
        word_split = no_bracket_t.split(' ')
        new_word_split = []
        for word in word_split:
            if not word.isalpha():
                continue
            in_there = 0
            for each in remove:
                if each in word:
                    in_there =1
                    break
            for each in danger:
                if each in word:
                    in_there =1
                    break
            if not in_there:
                new_word_split.append(word)
        if len(new_word_split)>70 and len(new_word_split)<100:
            new_txt.append(' '.join(new_word_split))
            new_titl.append(titls[i])
    return new_txt, new_titl

# Call the main Functions
main()
