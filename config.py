import numpy as np
import pandas as pd 
import re
import os 
import nltk
import math
import string 
import sklearn
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import * 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,make_scorer
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from gensim.models import Word2Vec
import pickle
import gensim
import csv
import logging
from joblib import dump,load


# config path
task1_path=""
task2_path="../task3/dataset/datasets_coursework1/Hateval/"
task3_path="../task3/dataset/datasets_coursework1/IMDb/"
## train
task3_train_pos = task3_path+"train/imdb_train_pos.txt"
task3_train_neg = task3_path+"train/imdb_train_neg.txt"
### dev
task3_dev_pos = task3_path+"dev/imdb_dev_pos.txt"
task3_dev_neg = task3_path+"dev/imdb_dev_neg.txt"
### test
task3_test_pos = task3_path+"test/imdb_test_pos.txt"
task3_test_neg = task3_path+"test/imdb_test_neg.txt"
### test_result
test_result_path = task3_path+"test/"


