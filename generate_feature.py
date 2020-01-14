from config import *

def generate_top_N_list(fi,fo) ->list:
    nl = [] # list which save top N words
    with open(fi,'r') as N:
        for words in N:
            nl.append(words.strip())
    print(nl)
    print(len(nl))
    return nl

# write the list to csv file 
def writeList2CSV(Mylist,filepath):
    with open(filepath,'w+') as fcsv:
        writer = csv.writer(fcsv)
        # write columns name 
        writer.writerow(nl100)
        for index in range(len(vector)):
            writer.writerow(vector[index])

################################# FEATURE 1 BAG OF WORDS ####################################
#generate  vectors  BAG OF WORDS,BOW #############
#### method 1 
def generate_bag_vector(corpus1,word_list) ->list:
    # word_list = Top N word
    # no label 
    bag_vector = [] # bag vector
    for index in range(len(corpus1)):
            vv = []
            for index_l in range(len(word_list)):
                if word_list[index_l] in corpus1[index]:
                    vv.append(str(corpus1[index].count(word_list[index_l])))
                else:
                    vv.append("0")
            vv.append(label)
            bag.append(vv)
    return bag_vector
#### method 2 
def generate_bag_vector2(corpus):
    # generate N-dimension vector
    vectorizer = CountVectorizer(max_features=1000,min_df=5,max_df=0.7,stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(corpus)
    X_array = X.toarray()
    #feature_name = vectorizer.get_feature_names()
    return X_array
################################# FEATURE 2 TF-IDF  ####################################
def generate_tf_idf_vector(corpus) ->list:
    vectorizer = CountVectorizer(max_features=1000,min_df=5,max_df=0.7,stop_words=stopwords.words('english'))
    transformer = TfidfTransformer()
    X = vectorizer.fit_transform(corpus)
    tfidf_vec = transformer.fit_transform(X).toarray()
    return tfidf_vec

################################# FEATURE 3 WORD2VEC ####################################
def generate_word2vector_model(corpus):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if os.path.exists('./model'):
        try:
            model = gensim.models.Word2Vec.load('./model')
            print("[info]: load model done")
        except Exception as e:
            print("[info]:load model faild,reason: ",e)
            model = Word2Vec(corpus,min_count=10,size=20,workers=4)
            model.save("./model")
    else:
        model = Word2Vec(corpus,min_count=10,size=20,workers=4)
        model.save("./model")
    return model 

def generate_word2vector(corpus,wvmodel):
    vec = []
    for index in range(len(corpus)):
        _vec = np.zeros(20).reshape((1,20))   
        count = 1
        for words in corpus[index]:
            try:
                count += 1 
                _vec += wvmodel[words].reshape((1,20))
                #print(_vec)
            except:
                pass
        _vec /= count
        vec.append(_vec)
    return vec
