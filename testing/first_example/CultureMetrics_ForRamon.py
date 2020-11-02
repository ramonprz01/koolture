
"""
The following algorithm performs the following steps:
    1. Run LDA on corpus of text
    2. Identify most salient topics from LDA output to interpret
    3. Extract prototypical text from corpus for topic interpretation 
    4. Show extract of prototypical text to reader

"""



'MODULES' 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
nltk.download('wordnet')
import scipy as sp
import math
import numpy as np
import csv
import matplotlib.pyplot as plt


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


' FUNCTIONS'
#read path
def read_path(path):
    import os
    file_list=os.listdir(path)
    return file_list


def jsd(p, q, base=np.e):#JS distance between probability vectors, used to compute compH
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    p = np.asarray(p)
    q = np.asarray(q)
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.


def conth(prob_matrix_df): # function to measure content heterogeneity given a topic (prob) matrix
    N = prob_matrix_df.shape[0]
    probMatrix = prob_matrix_df.values
    conth = 1/((sum(map(sum, np.square(probMatrix))))/N)
    return conth    


#COMPOSITIONAL HETEROGENEITY - CONSENSUS
#Interpretation: higher values indicate that employees characterize the firm
#using dissimilar topics (i.e. high comph = low consensus) 
def comph(probMatrix_df): 
    #Transform probMatrix_df into 2D array
    probMatrix = probMatrix_df.values

    x = 0
    y = 0
    
    df = pd.DataFrame()
    for x in range(0, len(probMatrix)): 
        jsd_list = []
        for y in range(0, len(probMatrix)): 
            jsd_list.append(jsd(probMatrix[x], probMatrix[y]))
            y = y+1
        df[str(x)] = jsd_list


    #Get df lower diagonal
    mask = np.ones(df.shape,dtype='bool')
    mask[np.triu_indices(len(df))] = False
    df_lower_diagonal = df[(df>-1)&mask]
    
    distance_list = []
    k = 0 
    for k in range(0, len(df)): 
    #Transform each column of df_lower_diagonal into list
        column_list = df_lower_diagonal[str(k)].values.tolist()
        #Drop nan values from column_list - to retain only actual values from lower diagonal 
        column_lower_diagonal_list = [l for l in column_list if (math.isnan(l) == False)]
        for d in column_lower_diagonal_list: 
            distance_list.append(d)
        k = k + 1
    comph = sum(distance_list) / float(len(distance_list))
    return comph





''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'Step 1: Update corpus of text to analyse via LDA'

'Isolate Netflix reviews from Glassdoor database & compute optimal nr of topics'
netflix_df = pd.read_pickle("Netflix_Data")

#select only on text of "pros", "cons" or "advice" from reviews, to run LDA separately
netflix_df = netflix_df.fillna('')
data_pros = netflix_df.pros.tolist()
data_pros_cleaned = []
for item in data_pros:
    if item != "":
        item = item.lower().replace("netflix", " ")
        item = item.replace("show less", "")
        item = item.replace("show more", "")
        item = item.replace("\n", "")
        item_modified =  ''.join([i for i in item if not i.isdigit()])
        data_pros_cleaned.append(item_modified)
        
#data_cons = netflix_df.cons.tolist()
#data_cons_cleaned = []
#for item in data_cons:
#    if item != "":
#        item = item.lower().replace("netflix", " ")
#        item = item.replace("show less", "")
#        item = item.replace("show more", "")
#        item = item.replace("\n", "")
#        item_modified =  ''.join([i for i in item if not i.isdigit()])
#        data_cons_cleaned.append(item_modified)

#data_advice = df_byFirm.advice.tolist()
#data_advice_cleaned = []
#for item in data_advice:
#    if item != "":
#        item = item.lower().replace("netflix", " ")
#        item = item.replace("show less", "")
#        item = item.replace("show more", "")
#        item = item.replace("\n", "")
#        item_modified =  ''.join([i for i in item if not i.isdigit()])
#        data_advice_cleaned.append(item_modified)




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'Step 2:Compute optimal number of topics on corpus'

#roughly counting total words included in the corpus
TotalWords_vectorizer = SBStemmedCountVectorizer(analyzer="word", stop_words='english')
#TotalWords_tf = TotalWords_vectorizer.fit_transform(data_pros_cleaned)
TotalWords_tf = TotalWords_vectorizer.fit_transform(data_cons_cleaned)
#TotalWords_tf = TotalWords_vectorizer.fit_transform(data_advice_cleaned)
totWords = len(TotalWords_vectorizer.get_feature_names())

#topic model
tf_vectorizer = SBStemmedCountVectorizer(max_df = 0.90, min_df=0.01, analyzer="word", stop_words='english')
#tf_vectorizer = LemmedCountVectorizer(max_df = 0.9, min_df=0.001, analyzer="word", stop_words='english')
#tf = tf_vectorizer.fit_transform(data_pros_cleaned)
tf = tf_vectorizer.fit_transform(data_cons_cleaned)
#tf = tf_vectorizer.fit_transform(data_advice_cleaned)

#extract list of all words included in the vocabulary
tf_feature_names = tf_vectorizer.get_feature_names()
#output vectorization of text corpus based on vocabulary
#tf_matrix = tf.toarray()

percVoc = len(tf_feature_names)/float(totWords)*100

i = 0    
output=np.zeros((60,3))

#totWordsPerdocument = np.sum(tf_matrix, axis=1)
for topics in range(2,300,5): 
    
    lda = LatentDirichletAllocation(n_components=topics, max_iter=200, learning_method='batch', learning_offset=10.,evaluate_every=2,random_state=1234)
    lda_fit = lda.fit(tf)
    #output normalized matrix with distributions of topics over words
    #normalized
    topicsOverWords = lda_fit.components_ / lda_fit.components_.sum(axis=1)[:, np.newaxis]
    topicsDissim_avg = comph(topicsOverWords)

#store results per firm   
    output[i,0] = topics
    output[i,1] = topicsDissim_avg 
    output[i,2] = percVoc
  
    i = i+1


#export results by firm
filename_save = ("TopicInterpretation_Netflix_Pros_OptimalTopics_Coherence")
#filename_save = ("TopicInterpretation_Netflix_Cons_OptimalTopics_Coherence")
#filename_save = ("TopicInterpretation_Netflix_Advice_OptimalTopics_Coherence")

results = open(filename_save + '.csv', 'w')

###creating csv file
writer = csv.writer(results)
writer.writerow(['glassdoorid', 'topics', 'coherence', 'voc%'])
for values in output:
    writer.writerow(values)
results.close()




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''Step 3: Run LDA on selected section of text reviews based on optimal topic number
generated above'''

#read file with coherence per topics
topics_coherence_df = pd.read_csv("TopicInterpretation_Netflix_Pros_OptimalTopics_Coherence.csv", sep=',')
#topics_coherence_df = pd.read_csv("TopicInterpretation_Netflix_Cons_OptimalTopics_Coherence.csv", sep=',')

#identify optimal topic number
optimal_topics = topics_coherence_df.topics[topics_coherence_df.coherence.idxmax()]



tf_vectorizer = SBStemmedCountVectorizer(max_df = 0.90, min_df=0.01, analyzer="word", stop_words='english')
#vectorize data (learn the vocabulary dictionary and return term-document matrix)
tf = tf_vectorizer.fit_transform(data_pros_cleaned)
#tf = tf_vectorizer.fit_transform(data_cons_cleaned)
#    extract features
tf_feature_names = tf_vectorizer.get_feature_names()
#for parameters of lda function - visit here: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
lda = LatentDirichletAllocation(n_components=int(optimal_topics), max_iter=200, learning_method='batch', learning_offset=10.,evaluate_every=2,random_state=1234)
lda_fit = lda.fit(tf)


#generate matrix summarizing distribution of docs (reviews) over topics
probMatrix = lda.transform(tf)
docs_topics_df = pd.DataFrame(data = probMatrix, index=None, columns=None, dtype=None, copy=False)




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'Step 4: Compute culture metrics

def topicModel(corpus, topics):
    tf_vectorizer = SBStemmedCountVectorizer(max_df = 0.90, min_df=0.01, analyzer="word", stop_words='english')
    #vectorize data (learn the vocabulary dictionary and return term-document matrix)
    tf = tf_vectorizer.fit_transform(corpus)
    
    #for parameters of lda function - visit here: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    lda = LatentDirichletAllocation(n_components=topics, 
                                max_iter=200, 
                                learning_method='batch', 
                                learning_offset=10.,
                                evaluate_every=2,
                                random_state=1234,
                                n_jobs=-1)
    #Fit lda model according to the given training data and parameters
    lda_fit = lda.fit(tf)
    
    #Output: Distribution of topics per document (project data to maximize class separation)
    probMatrix = lda.transform(tf)
    #Transform superCorpus_theta into pandas df
    probMatrix_df = pd.DataFrame(data = probMatrix, index=None, columns=None, dtype=None, copy=False)
    
    return probMatrix_df


docs_over_topics = topicModel(data_pros_cleaned, optimal_topics)
compH = comph(docs_over_topics)
contH = conth(docs_over_topics)

