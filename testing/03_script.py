import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk, re, math, csv
# nltk.download('wordnet')
# nlkt.download('punkt')

import koolture as kt
import time

from string import punctuation
from functools import partial
import concurrent.futures as cf
from collections import defaultdict



def main():
    df = pd.read_csv('../data/clean_gs.csv')

    ## Fist Range of Topics

    our_range = 2, 10, 50, 100, 150, 200, 250, 300

    comps_of_interest = df.employer.value_counts()

    comps_of_interest = (comps_of_interest[(comps_of_interest > 2)]).index

    cond2 = df['employer'].isin(comps_of_interest) # create the condition
    df_interest = df[cond2].copy() # get the new dataset
    unique_ids = df_interest['employer'].unique() # get the unique IDs or unique employers in the dataset

    reviews_nums = df_interest['employer'].value_counts().reset_index()
    reviews_nums.columns = ['company', 'reviews_nums']

    ## Fix Custom Stopwords List Before Cleaning

    data_pros = df_interest['pros'].values
    stopwords = nltk.corpus.stopwords.words('english') + [token.lower() for token in unique_ids]

    normalize_doc = partial(kt.normalize_doc, stopwords=stopwords)


    with cf.ProcessPoolExecutor() as e:
        data_pros_cleaned = e.map(normalize_doc, data_pros)
        data_pros_cleaned = list(e.map(kt.root_of_word, data_pros_cleaned))

    df_interest['pros_clean'] = data_pros_cleaned

    comp_ids = []
    i = 0

    for num in unique_ids:
        another_list = list(unique_ids[i:int(i+1000)])
        comp_ids.append(another_list)
        i += 1000

    chunks = list(filter(None, comp_ids))


    for i, comps_chunk in enumerate(chunks):

        start_time = time.time()

        cond2 = df_interest['employer'].isin(comps_chunk) # create the condition
        df_chunk = df_interest[cond2].copy() # get the new dataset
        unique_ids = df_chunk['employer'].unique() # get the unique IDs or unique employers in the dataset

        vectorizers_dicts = kt.get_vectorizers(data=df_chunk, unique_ids=unique_ids,
                                          company_col='employer', reviews_col='pros', 
                                          vrizer=CountVectorizer())

        reviews_nums = df_chunk['employer'].value_counts().reset_index()
        reviews_nums.columns = ['employerID', 'reviews_nums']

        partial_func = partial(kt.get_models, topics=our_range, 
                               vrizer_dicts=vectorizers_dicts, unique_ids=unique_ids)

        with cf.ProcessPoolExecutor() as e:
            output = list(e.map(partial_func, unique_ids))

        output_df = kt.build_dataframe(output)

        topics_sorted, comps, tops = kt.top_two_topics(data=output_df, companies_var='company',
                                       coherence_var='coherence', topics_var='topics',
                                       unique_ids=unique_ids, vrizers_list=vectorizers_dicts.values())

        partial_func = partial(kt.get_models, vrizer_dicts=vectorizers_dicts, unique_ids=unique_ids)

        with cf.ProcessPoolExecutor() as e:
            output2 = list(e.map(partial_func, comps, tops))

        output_df2 = kt.build_dataframe(output2)

        best_topics = kt.absolute_topics(output_df2, 'company', 'coherence', 
                                         'topics', 'models', vectorizers_dicts.values())


        docs_of_probas = defaultdict(pd.DataFrame)

        for tup in vectorizers_dicts.values():
            docs_of_probas[tup[0]] = pd.DataFrame(best_topics[tup[0]][1].transform(tup[1]))

        # Calculate the measures of interest

        comP_h_results = defaultdict(float)
        comT_h_results = defaultdict(float)
        entropy_avg_results = defaultdict(float)
        cross_entropy_results = defaultdict(float)

        for company, proba_df in docs_of_probas.items():
            comP_h_results[company] = kt.comph(proba_df.values)
            comT_h_results[company] = kt.conth(proba_df)
            entropy_avg_results[company] = kt.ent_avg(proba_df.values)
            cross_entropy_results[company] = kt.avg_crossEnt(proba_df.values)

        comph_df = pd.DataFrame.from_dict(comP_h_results.items())
        conth_df = pd.DataFrame.from_dict(comT_h_results.items())
        crossEnt_df = pd.DataFrame.from_dict(cross_entropy_results.items())
        cultureMetrics = comph_df.merge(conth_df, how = 'inner', right_on = 0, left_on = 0)
        cultureMetrics = cultureMetrics.merge(crossEnt_df, how = 'inner', right_on = 0, left_on = 0)
        cultureMetrics.columns = ['employerID', 'comph', 'conth', 'avgCrossEnt']

        df_best_topics = pd.DataFrame.from_records(best_topics).T.reset_index()
        df_best_topics.columns = ['employerID', 'best_topic', 'model', 'coherence']

        # df_best_topics.merge(reviews_nums, on='company', how='right').head()

        (cultureMetrics.merge(reviews_nums, on='employerID', how='right')
                       .merge(df_best_topics, on='employerID', how='right')
                       .to_csv(f'culturemetrics_chunk_num_{i}.csv', index=False))

        elapsed_time = time.time() - start_time
        print(f"Chunk # {i} just finished and it took: {str(elapsed_time)}")

    
    
    
if __name__ == '__main__':
    main()