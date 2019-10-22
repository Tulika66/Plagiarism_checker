#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import nltk
from nltk.corpus import stopwords


# In[268]:


import numpy as np
import pandas as pd
import copy
import hashlib
import string
import itertools


# In[5]:


###Preprocessing

stop_word = set(stopwords.words("english"))
punctuation_table = str.maketrans({key: " " for key in string.punctuation})
whitespace_table = str.maketrans({key: "" for key in string.whitespace})

def preprocess(document):
    
    document = document.casefold()
    document_split = (document.translate(punctuation_table)).split()
    document = ""
    for word in document_split:
        if word not in stop_word:
            document += word
    return document


# In[192]:



### Form a dictionary of shingles

cwd = os.getcwd()
doc_id = {}
shingles = {}
shingles_hash = {}
empty_shingle_presence = np.zeros(100)
k = 8
for i,file in enumerate(glob.glob(cwd + "/corpus-final09/*/*.txt")):
    document = preprocess((open(file, 'rb').read()).decode('utf-8','ignore'))
    doc_id[i] = file
    for index in range(len(document) - k + 1):
        shingle = document[index: index + k]
        if shingle not in shingles.keys():
            shingles[shingle] = copy.deepcopy(empty_shingle_presence)
        shingles[shingle][i] = 1
        shingles_hash[shingle] = int((hashlib.md5(shingle.encode())).hexdigest(), 16)


# In[263]:


len(shingles.keys())


# In[8]:


def find_perm(payload, xor):
    num1 = int((hashlib.md5(payload)).hexdigest(), 16)
    return num1 ^ xor


# In[276]:




### Find 100 permutations of the shingle
### Perform minhashing

hash_functions = 240
document_count = 100
signature_matrix = np.ones((hash_functions, document_count)) * np.inf

np.random.seed(101)
perm = np.arange(len(shingles.keys()))
for i in range(hash_functions):
    np.random.shuffle(perm)
    print(i, end = "\r")
    for j,key in enumerate(shingles.keys()):
        for k in range(document_count):
            if shingles[key][k] == 1 and signature_matrix[i][k] > perm[j]:
                signature_matrix[i][k] = perm[j]
    


# In[196]:


# %%time

# hash_functions = 240
# document_count = 100
# signature_matrix_new = np.ones((hash_functions, document_count)) * np.inf

# np.random.seed(101)
# for i in range(hash_functions):
#     print(i, end = "\r")
#     xor = int(np.random.rand() * pow(10,9))
#     for shingle in shingles.keys():
#         perm = shingles_hash[shingle] ^ xor
#         for j in range(document_count):
#             if shingles[shingle][j] == 1 and signature_matrix_new[i][j] > perm:
#                 signature_matrix_new[i][j] = perm


# In[277]:


signature_matrix.shape


# In[264]:


#signature_matrix.tofile("SignatureMatrixWith240Shingles=37142k=8")


# In[ ]:


# sm = np.fromfile("SignatureMatrixWith240Shingles=37142k=8")
# sm.reshape(240,100)


# In[235]:


def split_signature_matrix(signature_matrix, buckets, rows):
    divided_signature_matrix = []
    for i in range(0,signature_matrix.shape[0],rows):
        row = []
        for j in range(signature_matrix.shape[1]):
            element = signature_matrix[i:i+rows, j]
            row.append(element)
        divided_signature_matrix.append(np.asarray(row))
    return np.asarray(divided_signature_matrix)


# In[236]:


def find_jaccard_similarity(set1, set2):
    intersection = 0
    for i in range(len(set1)):
        if set1[i] == set2[i]:
            intersection += 1
    return intersection/len(set1)


# In[237]:


def find_euclidean_distance(set1, set2):
    dist = 0
    for i in range(set1):
        dist += (set1[i] - set2[i])**2
    return np.sqrt(dist)


# In[265]:


###Generate random vectors for cosine similarity of given array size

def form_random_vectors(size):
    x = []
    for j in [1,-1]:
        for i in itertools.repeat(j,size):
            x.append(i)
    vectors_set = set({})
    for i in itertools.permutations(x, size):
        vectors_set.add(i)
    random_vectors = []
    for v in vectors_set:
        vect = []
        for i in v:
            vect.append(i)
        random_vectors.append(vect)
    return np.asarray(random_vectors)


# In[321]:


def cosine_distance(set1, set2, random_vectors):
#     similarity = 0
#     for vect in random_vectors:
#         dot1 = (np.dot(set1, vect) > 0)
#         dot2 = (np.dot(set2, vect) > 0)
#         if dot1 == dot2:
#             similarity += 1
    len1 = 0
    len2 = 0
    for i in range(len(set1)):
        len1 += set1[i]**2
        len2 += set2[i]**2
    len_prod = (len1 * len2)**0.5
    return np.arccos((np.dot(set1, set2)) / len_prod) * 180/np.pi
#     return similarity/len(random_vectors)


# In[320]:


np.dot(signature_matrix_split[0,1], signature_matrix_split[0,2])


# In[365]:


get_ipython().run_cell_magic('time', '', '\n### Find Similarity \n### Define number of buckets and rows\n\nbuckets = 48\nrows = 5\nprint("rows = "  + str(rows))\njaccard_similarity_matrix = np.zeros((document_count, document_count))\nsignature_matrix_split = split_signature_matrix(signature_matrix, buckets, rows)\n\n### Jaccard Similarity\nfor i in range(document_count):\n    print(i, end = "\\r")\n    for j in range(document_count):\n        for k in range(buckets):\n            similar = find_jaccard_similarity(signature_matrix_split[k,i], signature_matrix_split[k,j])\n            jaccard_similarity_matrix[i][j] = max(jaccard_similarity_matrix[i][j], similar)\n            jaccard_similarity_matrix[j][i] = jaccard_similarity_matrix[i][j]\n\n### Cosine distance\n# random_vectors = form_random_vectors(rows)\n# cosine_similarity_matrix = np.ones((document_count, document_count)) * np.inf\n# for i in range(document_count):\n#     print(i, end = "\\r")\n#     for j in range(document_count):\n#         for k in range(buckets):\n#             distance = cosine_distance(signature_matrix_split[k,i], signature_matrix_split[k,j], random_vectors)\n#             cosine_similarity_matrix[i][j] = min(cosine_similarity_matrix[i][j], distance)\n#             cosine_similarity_matrix[j][i] = cosine_similarity_matrix[i][j]')


# In[349]:


setup_df = pd.read_excel("corpus-final09(copy).xls")
doc_plagiarism_type = {}
for i,file in enumerate(setup_df["File"]):
    doc_plagiarism_type[file] = setup_df.iloc[i,4]


# In[350]:


original_doc_id = []
for doc in doc_id.keys():
    if doc_id[doc][-14: -10] == "orig":
        original_doc_id.append(doc)


# In[351]:


tp = "True_Positives"
fp = "False_Positives"
tn = "True_Negative"
fn = "False_Negative"


# In[352]:


cut = "cut"
heavy = "heavy"
light = "light"
non = "non"


# In[369]:


##jaccard

similar_documents = {}
results = {}
performance = {}
results[tp] = results[fp] = results[fn] = results[tn] = results[cut] = results[heavy] = results[light] = results[non] = 0
threshold = 0.8

for k in original_doc_id:    
    similar_documents[ doc_id[k][-5] ] = []
    performance[ doc_id[k][-5] ] = copy.deepcopy(results) 
    
    for i in range(len( jaccard_similarity_matrix[k] )):
        if jaccard_similarity_matrix[k][i] >= threshold and k != i:
            similar_documents[ doc_id[k][-5] ].append(i)
            if doc_id[i][-5] == doc_id[k][-5]:
               # performance[doc_id[k][-5]][tp] += 1
               if doc_plagiarism_type[doc_id[i][-14:]]== non || doc_plagiarism_type[doc_id[i][-14:]]==heavy
                   performace[doc_id[k][-5]][fp] += 1
               performance[doc_id[k][-5]][doc_plagiarism_type[doc_id[i][-14:]]] += 1
            else:
               performace[doc_id[k][-5]][fp] += 1
  #  performance[doc_id[k][-5]][tn] = 19 - performance[doc_id[k][-5]][tp]
  #  performance[doc_id[k][-5]][fn] = 80 - performance[doc_id[k][-5]][fp]


# In[344]:


###Cosine

similar_documents = {}
results = {}
performance = {}
results[tp] = results[fp] = results[fn] = results[tn] = results[cut] = results[heavy] = results[light] = results[non] = 0
threshold = 15

for k in original_doc_id:    
    similar_documents[ doc_id[k][-5] ] = []
    performance[ doc_id[k][-5] ] = copy.deepcopy(results) 
    
    for i in range(len( jaccard_similarity_matrix[k] )):
        if cosine_similarity_matrix[k][i] <= threshold and k != i:
            similar_documents[ doc_id[k][-5] ].append(i)
            if doc_id[i][-5] == doc_id[k][-5]:
              #  performance[doc_id[k][-5]][tp] += 1
                performance[doc_id[k][-5]][doc_plagiarism_type[doc_id[i][-14:]]] += 1
            else:
                performace[doc_id[k][-5]][fp] += 1

print(performance)





