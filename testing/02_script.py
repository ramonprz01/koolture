# Culture Measures Based on Company Reviews

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk, re, math, csv
# nltk.download('wordnet')
# nlkt.download('punkt')

import koolture as kt

from string import punctuation
from functools import partial
import concurrent.futures as cf
from collections import defaultdict

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Load your dataset.

df = pd.read_csv('../data/clean_gs.csv')


# ## Fist Range of Topics

our_range = 2, 10, 50, 100, 150, 200, 250, 300


# Remove the company names from the reviews, and extract the reviews into a numpy array.

comps_of_interest = df.employer.value_counts()
comps_of_interest = (comps_of_interest[(comps_of_interest == 48)]).index
cond2 = df['employer'].isin(comps_of_interest) # create the condition
df_interest = df[cond2].copy() # get the new dataset
unique_ids = df_interest['employer'].unique() # get the unique IDs or unique employers in the dataset

reviews_nums = df_interest['employer'].value_counts().reset_index()
reviews_nums.columns = ['company', 'reviews_nums']


# ## Fix Custom Stopwords List Before Cleaning

# The text preprocessing of the corpus takes place in parallel. You first normalize the reviews and then take the root of the words.

data_pros = df_interest['pros'].values
stopwords = nltk.corpus.stopwords.words('english') + [token.lower() for token in unique_ids]
normalize_doc = partial(kt.normalize_doc, stopwords=stopwords)


# In[ ]:


with cf.ProcessPoolExecutor() as e:
    data_pros_cleaned = e.map(normalize_doc, data_pros)
    data_pros_cleaned = list(e.map(kt.root_of_word, data_pros_cleaned))
    

df_interest['pros_clean'] = data_pros_cleaned


# ## Create Vectorizers Tuple

vectorizers_dicts = kt.get_vectorizers(data=df_interest, unique_ids=unique_ids, company_col='employer', reviews_col='pros', vrizer=CountVectorizer())


# The following block run the models in parallel over the companies available and using the specifiedamount of topics in our_range variable and return a dictionary with the output of the get_models function for each company. It is used to identify the interval to search further for optimal topic number.


partial_func = partial(kt.get_models, topics=our_range, vrizer_dicts=vectorizers_dicts, unique_ids=unique_ids)

with cf.ProcessPoolExecutor() as e:
    output = list(e.map(partial_func, unique_ids))


# The next function will now iterate over the dictionary output from above, add each dataset into a list, and then concatenate them all into one dataset (output df contains exactly same information, but more readable, and used in next blocks).

output_df = kt.build_dataframe(output)


# The following loop iterates over the new dataframe, searches for the top 2 topics based on highest coherence, and appends to a list a tuple containing the company, a tuple with the top two topic numbers, and the fitted vectorizer from the original `vectorizers_list`.


topics_sorted, comps, tops = kt.top_two_topics(data=output_df, companies_var='company', coherence_var='coherence',
                                               topics_var='topics', unique_ids=unique_ids, vrizers_list=vectorizers_dicts.values())


partial_func = partial(kt.get_models, vrizer_dicts=vectorizers_dicts, unique_ids=unique_ids)

with cf.ProcessPoolExecutor() as e:
    output2 = list(e.map(partial_func, comps, tops))


# Create multiple dataframes from dictionaries again and collapse them into 1.

output_df2 = kt.build_dataframe(output2)

# Search for the best topic based on the new output, and get the top 10 words per topic. At the moment, you are only adding 1 of the topics for each company but you can change this by removing the indexing in `top_topics` below.

best_topics = kt.absolute_topics(output_df2, 'company', 'coherence', 'topics', 'models', vectorizers_dicts.values())


# Check out your output.

# Get the probabilities dataframes for each company and add them to a dictionary.

# In[ ]:


#generate matrix summarizing distribution of docs (reviews) over topics
docs_of_probas = defaultdict(pd.DataFrame)

for tup in vectorizers_dicts.values():
    docs_of_probas[tup[0]] = pd.DataFrame(best_topics[tup[0]][1].transform(tup[1]))


# # Calculate the measures of interest


comP_h_results = defaultdict(float)
comT_h_results = defaultdict(float)
entropy_avg_results = defaultdict(float)
cross_entropy_results = defaultdict(float)
for company, proba_df in docs_of_probas.items():
    comP_h_results[company] = kt.comph(proba_df.values)
    comT_h_results[company] = kt.conth(proba_df)
    entropy_avg_results[company] = kt.ent_avg(proba_df.values)
    cross_entropy_results[company] = kt.avg_crossEnt(proba_df.values)')


comph_df = pd.DataFrame.from_dict(comP_h_results.items())
conth_df = pd.DataFrame.from_dict(comT_h_results.items())
crossEnt_df = pd.DataFrame.from_dict(cross_entropy_results.items())
cultureMetrics = comph_df.merge(conth_df, how = 'inner', right_on = 0, left_on = 0)
cultureMetrics = cultureMetrics.merge(crossEnt_df, how = 'inner', right_on = 0, left_on = 0)
cultureMetrics.columns = ['employerID', 'comph', 'conth', 'avgCrossEnt']
cultureMetrics.head()


df_best_topics = pd.DataFrame.from_records(best_topics).T.reset_index()
df_best_topics.columns = ['company', 'best_topic', 'model']
df_best_topics.head()

cultureMetrics.to_csv('CultureMetrics_TestSample_1000.csv', index=False)
(df_best_topics.merge(reviews_nums, on='company', how='right')
               .to_csv('best_topics.csv', index=False))

