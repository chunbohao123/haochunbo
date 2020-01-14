from config import * 
from generate_feature import *
from data_preprocess import *

def classification_report_with_accuracy_score(y_ture,y_pred):
    print(classification_report(y_ture,y_pred))
    return accuracy_score(y_ture,y_pred)

def svm_model(train,tag,test=None,name=None):
    #data = pd.read_csv(fi,index_col=0)
    #print(data.shape)
    #x = data.drop('label',axis=1)
    #y = data['label']
    print("[info]: len(train): {0}, len(tag): {1}".format(train,tag))
    assert len(train)==len(tag),"len(train) != len(tag)"
    ## split train test data
    x_train,x_test,y_train,y_test = train_test_split(np.array(train,dtype='float64'),np.array(tag,dtype='float64'),test_size =0.3,random_state=0)

    # train model 
    clf = SVC() # kernel = rbf 
    #clf = SVC(kernel='poly',degree=8)
    clf.fit(x_train,y_train)
    dump(clf,name+"svm_model.joblib")
    # predict
    y_pred = clf.predict(x_test)

    # evaluting the algorithm
    print("[info]:confusion_matrix:")
    print(confusion_matrix(y_test,y_pred))
    print("[info]:classification_report:")
    print(classification_report(y_test,y_pred))

def predict_text(test,tag,name=None):
    print("[info]:loading predict model.... please wait a moment")
    clf = load(name+"svm_model.joblib")
    y_pred = clf.predict(test)
    print("[info]:confusion_matrix:")
    print(confusion_matrix(tag,y_pred))
    print("[info]:classification_report:")
    print(classification_report(tag,y_pred))
    with open(test_result_path+name+"_result.log",'w+') as fws:
        fws.write("predict value ")
        fws.write("true value ")
        fws.write("\n")
        for index in range(len(y_pred)):
            fws.write(str(y_pred[index]))
            fws.write("\t")
            fws.write(str(tag[index]))
            fws.write("\n")
    print("[info]:generating predict result done")
    print("[info]:save path:",test_result_path+name+"_result.log")
if "name == __main__":

    ##############################training###################################

    ################################step 1 get corpus#########################
    print("[info]:geting corpus.... please wait for a moment")
    corpus_train_pos = get_corpus(task3_train_pos)
    corpus_train_neg = get_corpus(task3_train_neg)
    all_corpus = get_all_corpus(task3_train_pos,task3_train_neg)
    print("[info]:geting corpus done")
    ##############################step 2 generate vectors ####################
    ################################ bag featrues#############################
    print("[info]:Generating Bag features... please wait for a moment,it may take hours~~")
    bag_vec_train_pos = generate_bag_vector2(corpus_train_pos)
    bag_vec_train_neg = generate_bag_vector2(corpus_train_neg)
    bag_vec_tag_pos = np.zeros(len(bag_vec_train_pos))
    bag_vec_tag_neg = np.ones(len(bag_vec_train_neg))
    bag_vec_train = np.concatenate((bag_vec_train_pos,bag_vec_train_neg))
    bag_vec_train_tag = np.concatenate((bag_vec_tag_pos,bag_vec_tag_neg))
    print("[info]:Generating Bag done")
    print("[info]:X shape: {0}, Y shape: {1}".format(bag_vec_train.shape,bag_vec_train_tag.shape))
    #print(type(bag_vec))
    print("[info]:training svm model using bag feature.... please wait ")
    svm_model(bag_vec_train,bag_vec_train_tag,"","bag")
    print("[info]:training svm model using bag feature done ")
    #clear the vector
    bag_vec_train_pos = None
    bag_vec_train_neg = None
    bag_vec_tag_neg = None
    bag_vec_tag_pos = None
    bag_vec_train=None
    bag_vec_train_tag = None
    ################################tf_idf###################################
    print("[info]:Generating tf-idf features... please wait for a moment,it may take hours~~")
    tf_idf_train_pos = generate_tf_idf_vector(corpus_train_pos)
    tf_idf_train_neg = generate_tf_idf_vector(corpus_train_neg)
    tf_idf_tag_pos = np.zeros(len(tf_idf_train_pos))
    tf_idf_tag_neg = np.ones(len(tf_idf_train_neg))
    tf_idf_train =np.concatenate((tf_idf_train_pos,tf_idf_train_neg))
    tf_idf_tag = np.concatenate((tf_idf_tag_pos,tf_idf_tag_neg))
    print("[info]:Generating tf-idf features... please wait for a moment")
    print("[info]:training svm model using tf-idf  feature.... please wait ")
    svm_model(tf_idf_train,tf_idf_tag,"","tf_idf")
    print("[info]:training svm model using tf-idf feature done ")
    # clear the vector 
    tf_idf_train_pos = None
    tf_idf_train_neg = None
    tf_idf_tag_neg = None
    tf_idf_tag_pos = None
    tf_idf_train = None
    tf_idf_tag = None
    ################################ word2vec ###################################
    print("[info]:Generating word2vec features... please wait for a moment,it may take hours~~")
    word2vec_vec_model = generate_word2vector_model(all_corpus)
    all_corpus=None
    wvx_train_pos = generate_word2vector(corpus_train_pos,word2vec_vec_model)
    corpus_train_pos = None
    print("[info]:Generating word2vec pos features done ")
    wvx_train_neg = generate_word2vector(corpus_train_neg,word2vec_vec_model)
    corpus_train_neg = None
    print("[info]:Generating word2vec neg features done ")
    pos_tag = np.zeros(len(wvx_train_pos))
    neg_tag = np.ones(len(wvx_train_neg))
    wvx_train = wvx_train_pos + wxv_train_neg
    train_tag = np.concatenate((pos_tag,neg_tag))
    print("[info]:Generating word2vec done ")
    print("[info]:training svm model using word2vec feature.... please wait ")
    svm_model(wvx_train,train_tag,"","word2vec")
    print("[info]:training svm model using word2vec feature done ")

    print("[info]:starting predict the text.....")
    # clear the vector
    wvx_train_pos = None
    wvx_train_neg = None
    neg_tag = None
    pos_tag = None
    wvx_train = None
    train_tag = None
    ###########################predicting#######################################
    # clear train corpus
    corpus_train_pos = None
    corpus_train_neg = None
    all_corpus = None
    print("[info]:start predicting ...")

    corpus_test_pos = get_corpus(task3_test_pos)
    corpus_test_neg = get_corpus(task3_test_neg)
    print("[info]:generating test bag vector.... please wait")
    # bag vector 
    bag_vec_test_pos = generate_bag_vector2(corpus_test_pos)
    bag_vec_test_neg = generate_bag_vector2(corpus_test_neg)
    bag_vec_tag_pos = np.zeros(len(bag_vec_test_pos))
    bag_vec_tag_neg = np.ones(len(bag_vec_test_neg))
    bag_vec_test = np.concatenate((bag_vec_test_pos,bag_vec_test_neg))
    bag_vec_test_tag =np.concatenate((bag_vec_tag_pos,bag_vec_tag_neg))
    print("[info]:Generating Bag test vector  done ")
    #print(type(bag_vec))
    print("[info]:predicting.... please wait ")
    predict_text(bag_vec_test,bag_vec_test_tag,"bag")
    bag_vec_test_pos = None
    bag_vec_test_neg = None
    bag_vec_tag_neg = None
    bag_vec_tag_pos = None
    bag_vec_test = None
    bag_vec_test_tag = None
    #tf-idf vector
    print("[info]:Generating test tf-idf features... please wait for a moment")
    tf_idf_test_pos = generate_tf_idf_vector(corpus_test_pos)
    tf_idf_test_neg = generate_tf_idf_vector(corpus_test_neg)
    tf_idf_tag_pos = np.zeros(len(tf_idf_test_pos))
    tf_idf_tag_neg = np.ones(len(tf_idf_test_neg))
    tf_idf_test =np.concatenate((tf_idf_test_pos,tf_idf_test_neg))
    tf_idf_tag =np.concatenate((tf_idf_tag_pos,tf_idf_tag_neg))
    print("[info]:Generating test tf-idf features done ")
    print("[info]:predicting.... please wait ")
    predict_text(tf_idf_test,tf_idf_tag,"tf_idf")
    print("[info]:testing svm model using tf-idf feature done ")
    tf_idf_test_pos =None
    tf_idf_test_neg =None
    tf_idf_tag_neg =None
    tf_idf_tag_pos =None
    tf_idf_test =None
    tf_idf_tag =None
    # word2vec
    print("[info]:Generating test word2vec features... please wait for a moment")
    wvx_test_pos = generate_word2vector(corpus_test_pos,word2vec_vec_model)
    print("[info]:Generating test word2vec pos features done ")
    wvx_test_neg = generate_word2vector(corpus_test_neg,word2vec_vec_model)
    print("[info]:Generating test word2vec neg features done ")
    pos_tag = np.zeros(len(wvx_test_pos))
    neg_tag = np.ones(len(wvx_test_neg))
    wvx_test = wvx_test_pos + wxv_test_neg
    pos_tag_l = list(pos_tag)
    neg_tag_l = list(neg_tag)
    test_tag = pos_tag_l + neg_tag_l
    print("[info]:Generating test word2vec features done ")
    print("[info]:predicting.... please wait ")
    predict_text(wvx_test,test_tag,"word2vec")
    print("[info]:testing svm model using tf-idf feature done ")
 
