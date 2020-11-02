def comph(probMatrix_df, arr_or_df='df'): 
    #Transform probMatrix_df into 2D array
    
    """
    On average how similar are people on reviews over topics.
    """
    
    
    if arr_or_df == 'df':
        probMatrix = probMatrix_df
    else:
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
    mask = np.ones(df.shape, dtype='bool')
    mask[np.triu_indices(len(df))] = False
    df_lower_diagonal = df[(df>-1) & mask]
    
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




def root_of_word(docs, root_word_method='lemma'):
    
    porter_stemmer = nltk.stem.PorterStemmer()
    snowball_stemmer = nltk.stem.SnowballStemmer('english')
    lemma = nltk.wordnet.WordNetLemmatizer()
    
    for text in docs:
        
    
        tokens = nltk.word_tokenize(text)

        if root_word_method == 'lemma':
            doc = ' '.join([lemma.lemmatize(w) for w in tokens])
        elif root_word_method == 'stemm':
            doc = ' '.join([porter_stemmer.stem(w) for w in tokens])
        elif root_word_method == 'snowball':
            doc = ' '.join([snowball_stemmer.stem(w) for w in tokens])
        
    return doc

stemming = partial(root_of_word, root_word_method='stemm')
snowball = partial(root_of_word, root_word_method='snowball')




### my comph


def comph(probMatrix_df, arr_or_df='df'): 
    #Transform probMatrix_df into 2D array
    
    """
    On average how similar are people on reviews over topics.
    """
    if arr_or_df == 'df':
        probMatrix = probMatrix_df
    else:
        probMatrix = probMatrix_df.values

    
    df = pd.DataFrame()
    for x in range(len(probMatrix)): 
        jsd_list = []
        for y in range(len(probMatrix)): 
            jsd_list.append(jsd(probMatrix[x], probMatrix[y]))
        df[str(x)] = jsd_list


    #Get df lower diagonal
    mask = np.ones(df.shape, dtype='bool')
    mask[np.triu_indices(len(df))] = False
    df_lower_diagonal = df[(df>-1) & mask]
    
    distance_list = []
    for k in df.columns: 
    #Transform each column of df_lower_diagonal into list
        col_array = df_lower_diagonal.loc[df_lower_diagonal[k].notna(), k].values
        for d in col_array:
            distance_list.append(d)

    return (sum(distance_list) / len(distance_list))

####### my conth

def conth(p_mtx_df): # function to measure content heterogeneity given a topic (prob) matrix
    '''
    How is this review spread across the topics.
    Then you take the average values across the reviews
    Herfindall index
    Assuming the reviews are about culture
    Are people on average focus on a few cultural values (topics) when they write their review
    '''
    return (1 / ((sum(map(sum, np.square(p_mtx_df.values)))) / p_mtx_df.shape[0]))


####### the pipeline

pipe = Pipeline([
    ('normalizer', FunctionTransformer(corp_normalizer)),
    ('transformer', FunctionTransformer(root_of_word)),
    ('vectorizer', CountVectorizer(max_df = 0.90, min_df=0.01, stop_words='english')),
    ('lda', LatentDirichletAllocation(n_components=10, max_iter=200, evaluate_every=2, random_state=42))
])



%%time


output = defaultdict(np.float32)

def mapping_topics(topics):
    
    lda = LatentDirichletAllocation(n_components=topics, max_iter=200, learning_method='batch', 
                                    learning_offset=10., evaluate_every=2, random_state=1234)#, n_jobs=-1)
    ldamo = lda.fit(tf)
    
    return ldamo

    #output normalized matrix with distributions of topics over words
    #normalized
#     topicsOverWords = ldamo.components_ / ldamo.components_.sum(axis=1)[:, np.newaxis]
#     topicsDissim_avg = comph(topicsOverWords)
    
#     output = (topics, topicsDissim_avg, percVoc)
#     return output