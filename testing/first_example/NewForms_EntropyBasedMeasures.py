#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:12:09 2020

@author: ariannamarchetti
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
import scipy as sp
import math
from math import log2
import statistics

def create_alpha_vector(A, beta):
#Columns have - on average - the same probability
#    All 1s
    alpha_vector = beta*np.ones(A, dtype=np.int)
#Completely random probabilities per vector    
#    alpha_vector = np.random.choice(int(10),(A),replace=True)+1
    return alpha_vector
#Add mode



def jsd(p, q, base=np.e):#JS distance between probability vectors, used to compute compH
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    p = np.asarray(p)
    q = np.asarray(q)
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.

#CONTENT HETEROGENEITY - SHARPNESS
#Interpretation: higher values indicate that employees discuss a broader range of cultural
#topics, while lower values indicate a narrower, concentrated set of topics 
#(i.e. low conth = high sharpness)
# def conth(probMatrix): # function to measure content heterogeneity given a topic (prob) matrix
#     conth = (sum(map(sum, np.square(probMatrix))))/N
#     return conth

def conth(probMatrix): # function to measure content heterogeneity given a topic (prob) matrix
    return ((sum(map(sum, np.square(probMatrix))))/N)


#COMPOSITIONAL HETEROGENEITY - CONSENSUS
#Interpretation: higher values indicate that employees characterize the firm
#using dissimilar topics (i.e. high comph = low consensus) 
def comph(probMatrix):
    #Transform probMatrix_df into 2D array

    i = 0
    j = 0
    
    df = pd.DataFrame()
    for i in range(0, len(probMatrix)): 
        jsd_list = []
        for j in range(0, len(probMatrix)): 
            jsd_list.append(jsd(probMatrix[i], probMatrix[j]))
            j = j+1
        df[str(i)] = jsd_list


    #Get df lower diagonal
    mask = np.ones(df.shape,dtype='bool')
    mask[np.triu_indices(len(df))] = False
    df_lower_diagonal = df[(df>-1)&mask]
    
    distance_list = []
    i = 0 
    for i in range(0, len(df)): 
    #Transform each column of df_lower_diagonal into list
        column_list = df_lower_diagonal[str(i)].values.tolist()
        #Drop nan values from column_list - to retain only actual values from lower diagonal 
        column_lower_diagonal_list = [x for x in column_list if (math.isnan(x) == False)]
        for d in column_lower_diagonal_list: 
            distance_list.append(d)
        i = i + 1
    comph = 1-(sum(distance_list) / float(len(distance_list)))
    return comph


#function to measure the avg entropy of a probability distribution
def ent_avg(probMatrix): 
    import statistics
    entropy_list = []
    for i in range(len(probMatrix)): 
        entropy_list.append(entropy(probMatrix[i]))
    entropy_avg = statistics.mean(entropy_list)
    return entropy_avg    

# function to compute the cross-entropy of two probability distributions
def cross_entropy(p, q):
    for i in range(len(p)):
        p[i] = p[i]+1e-12
    for i in range(len(q)):
        q[i] = q[i]+1e-12

    return -sum([p[i]*log2(q[i]) for i in range(len(p))])


# function to compute the average cross-entropy of a matrix
def avg_crossEnt(probMatrix): 
#    NOTE: Cross entropy is not symmetric. 
#    This function takes both cross-entropy(p,q) and cross-entropy(q,p) 
#    into account when computing the avg
    crossEntropy_list = []
    for i in range(len(probMatrix)):
        for j in range(len(probMatrix)): 
            if i != j:
                crossEntropy_list.append(cross_entropy(probMatrix[i], probMatrix[j]))
    crossEntropy_avg = statistics.mean(crossEntropy_list)
    return crossEntropy_avg    


#parameters to tune
I=50# number of iterations to run the simulation
A=10#topics
N=50 #number of agents
beta=0.01#parameter to tune the Dirichlet distribution


' CURRENT METRICS FOR FOCUS AND SIMILARITY'

output_final=np.zeros((0,2))

i = 0
for i in range (I): 
    topics = np.random.dirichlet(create_alpha_vector(A, beta), N)
    output = np.zeros((1,2))      
    output[0,0] = comph(topics)
    output[0,1] = conth(topics)
    output_final = np.append(output_final, output, axis = 0) 
    i = i + 1

output_final_df = pd.DataFrame(output_final, columns = ['comph', 'conth'])
correlationMatrix = output_final_df.corr(method='pearson')






' ENTROPY-BASED MEASURES'


output_final_ent=np.zeros((0,2))

i = 0
for i in range (I): 
    topics = np.random.dirichlet(create_alpha_vector(A, beta), N)
    output = np.zeros((1,2))      
    output[0,0] = ent_avg(topics)
    output[0,1] = avg_crossEnt(topics)
    output_final_ent = np.append(output_final_ent, output, axis = 0) 
    i = i + 1

output_final_ent_df = pd.DataFrame(output_final_ent, columns = ['ent_avg', 'crossEnt_avg'])
correlationMatrix_ent = output_final_ent_df.corr(method='pearson')












''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

' APPLICATION TO LH AND T SAMPLES'





import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
nltk.download('wordnet')




'CLASSES'

#define classes for different types of stemming / lemmatization
#for difference between stemming and lemmatizing: 
#https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
#however, based on testing, I am preferring stemming so far
porter_stemmer = nltk.stem.PorterStemmer()
class PStemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(PStemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([porter_stemmer.stem(w) for w in analyzer(doc)])

snowball_stemmer = nltk.stem.SnowballStemmer('english')
class SBStemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(SBStemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([snowball_stemmer.stem(w) for w in analyzer(doc)])


lemma = nltk.wordnet.WordNetLemmatizer()
class LemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([lemma.lemmatize(w) for w in analyzer(doc)])

#examples of using the three approaches - porter and snowball seems the same
#
#lemma.lemmatize('articles')
#'article'
#lemma.lemmatize('leaves')
#'leaf'
#lemma.lemmatize('abundance')
#'abundance'
#lemma.lemmatize('abundant')
#'abundant'
#
#porter_stemmer.stem('articles')
#'articl'
#porter_stemmer.stem('leaves')
#'leav'
#porter_stemmer.stem('abundance')
#'abund'
#porter_stemmer.stem('abundant')
#'abund'
#
#snowball_stemmer.stem('articles')
#'articl'
#snowball_stemmer.stem('leaves')
#'leav'
#snowball_stemmer.stem('abundance')
#'abund'
#snowball_stemmer.stem('abundant')
#'abund'


'FUNCTIONS'
#read path
def read_path(path):
    import os
    file_list=os.listdir(path)
    return file_list

#function read corpus as list of text documents, and returns df with probability
#of documents over topics (matrix of thetas over topics)
def topicModel(corpus, topics):
    tf_vectorizer = SBStemmedCountVectorizer(max_df = 0.90, min_df=0.01, analyzer="word", stop_words='english')
    #vectorize data (learn the vocabulary dictionary and return term-document matrix)
    tf = tf_vectorizer.fit_transform(corpus)
    
    #for parameters of lda function - visit here: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    lda = LatentDirichletAllocation(n_components=topics, max_iter=200, learning_method='batch', learning_offset=10.,evaluate_every=2,random_state=1234)
    #Fit lda model according to the given training data and parameters
    lda_fit = lda.fit(tf)

    
    #Output: Distribution of topics per document (project data to maximize class separation)
    probMatrix = lda.transform(tf)
    #Transform superCorpus_theta into pandas df
    probMatrix_df = pd.DataFrame(data = probMatrix, index=None, columns=None, dtype=None, copy=False)
    return probMatrix_df



#function to measure the avg entropy of a probability distribution
def ent_avg(probMatrix_df): 
    import statistics
    probMatrix = probMatrix_df.values
    entropy_list = []
    for i in range(len(probMatrix)): 
        entropy_list.append(entropy(probMatrix[i]))
    entropy_avg = statistics.mean(entropy_list)
    return entropy_avg    




#sample I
    
#LH firms (all the docs_topics matrixes are already available)

#read location of docs_topics matrices  
file_list_nf = read_path(r"/Users/ariannamarchetti/Dropbox/Ari/PhD_Dropbox/Year 4/Research Projects/Essay 3/Arianna/New Forms/Python_DF/LH_TopicModels/")
path_nf = r"/Users/ariannamarchetti/Dropbox/Ari/PhD_Dropbox/Year 4/Research Projects/Essay 3/Arianna/New Forms/Python_DF/LH_TopicModels/"



#compute entropy and save results
output_final_nf = np.empty((0,2))
for fileName in file_list_nf: 
    if fileName != '.DS_Store':
        path_final = path_nf + fileName
        docs_topics_df =  pd.read_pickle(path_final)

        output=np.zeros((1,2))
        output[0,0] = int(fileName.split('_')[2])
        output[0,1] = ent_avg(docs_topics_df)
        #append results to total output - across firms
        output_final_nf = np.append(output_final_nf, output, axis = 0)  

nf_entropy = pd.DataFrame(output_final_nf, columns=['glassdoorid_x','avgEntropy'])

nf_entropy.to_excel("NewForms_LH_AvgEntropy.xlsx")




' T firms ' 

#1. compute and save missing topic models

#upload xls containing info about T firms for which docs_over_topics are missing
ids_df = pd.read_excel('/Users/ariannamarchetti/Dropbox/Ari/PhD_Dropbox/Year 4/Research Projects/Essay 3/Arianna/New Forms/NewForms_Shared_AriannaPhanish/KeepsDying/DataForAnalyses/NewForms_T_MissingIDs_ForEntropy.xlsx')
ids_list = ids_df.comp_id.to_list()

#import Glassdoor data fpr T firms
file_list_comp = read_path(r"/Users/ariannamarchetti/Dropbox/NonHierarchies_DF/MatchingSample/BestMatchingSample_Group1To4//")
path_comp = r"/Users/ariannamarchetti/Dropbox/NonHierarchies_DF/MatchingSample/BestMatchingSample_Group1To4//"

#import data about culture metrics (included optimal number of topics) for T firms
#Pros
path_m = "/Users/ariannamarchetti/Dropbox/Ari/PhD_Dropbox/Year 4/Research Projects/Essay 3/Arianna/New Forms/CultureMetrics/Competitors/Pros/NewForms_Pros_CultureMetrics_Comp.csv"
culture_m = pd.read_csv(path_m, sep=',')


for compID in ids_list: 
#    upload Glassdoor reviews

    for fileName in file_list_comp: 
        if fileName != '.DS_Store':
            if compID == int(fileName.split("_")[0]): 
                path_final = path_comp + fileName
                df_byFirm =  pd.read_pickle(path_final)


                #prepare data for topic model 
                df_byFirm = df_byFirm.fillna('')
                #to ensure that df and data_pros have same dimension, used to compute culture metrics
                #at firm-level in next step
                df_byFirm = df_byFirm.drop(df_byFirm.index[df_byFirm.pros == ''])
                df_byFirm = df_byFirm.reset_index()
                df_byFirm = df_byFirm.drop(columns = ['index'])

                data_pros = df_byFirm.pros.tolist()
                data_pros_cleaned = []
                for item in data_pros:
                    if item != "":
                        item = item.lower()
                        item = item.replace("show less", "")
                        item = item.replace("\n", "")
                        item_modified =  ''.join([i for i in item if not i.isdigit()])
                        data_pros_cleaned.append(item_modified)

    
#                identify optimal topic number
                optimal_topics = int(culture_m.loc[culture_m.glassdoorid == compID].optimal_topics[culture_m.loc[culture_m.glassdoorid == compID].index[0]])

                    #computed in python script "AllFirms_OptimalTopicsNum_PuranamEtAl_2018"
    
#            run LDA
                docs_over_topics = topicModel(data_pros_cleaned, optimal_topics)

                #save docs_topics in df format
                docs_over_topics.to_pickle("/Users/ariannamarchetti/Dropbox/Ari/PhD_Dropbox/Year 4/Research Projects/Essay 3/Arianna/New Forms/Python_DF/T_TopicModels_SampleI/T_TopicModel_"+str(compID))

   


#2. Compute entropy

#read location of docs_topics matrices  
file_list_comp = read_path(r"/Users/ariannamarchetti/Dropbox/Ari/PhD_Dropbox/Year 4/Research Projects/Essay 3/Arianna/New Forms/Python_DF/T_TopicModels_SampleI/")
path_comp = r"/Users/ariannamarchetti/Dropbox/Ari/PhD_Dropbox/Year 4/Research Projects/Essay 3/Arianna/New Forms/Python_DF/T_TopicModels_SampleI/"



#compute entropy and save results
output_final_comp = np.empty((0,2))
for fileName in file_list_comp: 
    if fileName != '.DS_Store':
        path_final = path_comp + fileName
        docs_topics_df =  pd.read_pickle(path_final)

        output=np.zeros((1,2))
        output[0,0] = int(fileName.split('_')[2])
        output[0,1] = ent_avg(docs_topics_df)
        #append results to total output - across firms
        output_final_comp = np.append(output_final_comp, output, axis = 0)  

comp_entropy = pd.DataFrame(output_final_comp, columns=['glassdoorid_x','avgEntropy'])
comp_entropy.to_excel("NewForms_T_AvgEntropy.xlsx")













