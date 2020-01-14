from config import *
################################ read the text  ##################################################################
def read_csv(fp):
    # fp : file path
    column_names = ["id","text","label"]
    #parsed = pd.read_csv(fp,skiprows = 1,names=column_names,sep='\t',nrows=10,engine='python')
    parsed = pd.read_csv(fp,skiprows = 1,names=column_names,sep='\t',nrows=79,engine='python')
    print(parsed)
'''
def split_file(fp):
    fr = open(fp,'r')
    f_pos = open("pos.log",'w+')
    f_neg = open("neg.log",'w+')
    for lines in fr:
        data = lines.strip().split("\t")
        if len(data) < 2:
            print("[ERROR]: split error")
            break
        if data[2]is '1':
            print("[INFO]: neg label: ",data[2])
            string = re_at.sub("",data[1])
            string = re_http.sub("",string)
            string = re_null.sub(" ",string)
            string = re_dot.sub(" ",string)
            string = re_q.sub(" ",string)
            string = re_space.sub(" ",string)
            f_neg.write(string.lower())
            f_neg.write("\n")
        if data[2] is '0':
            print("[INFO]: pos label: ",data[2])
            string = re_at.sub("",data[1])
            string = re_http.sub("",data[1])
            string = re_dot.sub(" ",string)
            string = re_q.sub(" ",string)
            string = re_space.sub(" ",string)
            string = re_null.sub(" ",string)
            f_pos.write(string)
            f_pos.write("\n")

'''
def get_tokens(text):
    lowers = text.lower()
    remove_punctuation_map = dict((ord(char),None)for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens
# count
def Count(token):
    count = Counter(token)
    print ("TOP 10 is {}".format(count.most_common(10)))
    return count
# get stem
def stem_tokens(tokens,stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
# stop words
def stop_words(text):
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    count = Counter(filtered)
    print("TOP 10 works",count.most_common(10))
    return count
# get stem & stop words 
def get_stem(text):
    tokens = get_tokens(text)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed = stem_tokens(filtered, stemmer)
    count = Counter(stemmed)
    return count 

def get_corpus(fi)-> list:
    corpus = []
    _temp = ""
    with open (fi,'r') as fr:
        for lines in fr:
            tokens = get_tokens(lines)
            for index in range(len(tokens)):
                _temp = _temp + tokens[index]
                _temp = _temp + " "
            #print(_temp)
            corpus.append(_temp)
    return corpus

def get_all_corpus(fi1,fi2)-> list:
    all_corpus = []
    with open (fi1,'r') as fr:
        for lines in fr:
            tokens = get_tokens(lines)
            _temp = ""
            for index in range(len(tokens)):
                _temp = _temp + tokens[index]
                _temp = _temp + " "
            #print(_temp)
            all_corpus.append(_temp)

    with open (fi2,'r') as fr:
        for lines in fr:
            tokens = get_tokens(lines)
            _temp = ""
            for index in range(len(tokens)):
                _temp = _temp + tokens[index]
                _temp = _temp + " "
            #print(_temp)
            all_corpus.append(_temp)

    return all_corpus


#if "name == __main__":
#    pass
#    vectorizer = CountVectorizer()
#    transformer = TfidfTransformer()
#    tfidf = transformer.fit_transform(vectorizer.fit_transform(get_corpus(task3_train_pos)))
#    print(tfidf)
