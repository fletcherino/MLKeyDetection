from flask import Flask, render_template,request
import pymysql
import credentials
import pandas as pd
import matplotlib.pyplot as plt


# import seaborn as sns

# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)


class Database:

    def __init__(self):
        host = credentials.COMSC_MYSQL
        user = credentials.USERNAME
        password = credentials.PASSWORD
        db = credentials.DATABASE
        self.con = pymysql.connect(host=host, user=user, password=password, db=db, cursorclass=pymysql.cursors.
                                   DictCursor)
        self.cur = self.con.cursor()

    def list_tones(self, topn=1000):
        self.cur.execute("SELECT tone1,tone2,tone3,tone_key FROM tones LIMIT %d" % (topn))
        result = self.cur.fetchall()
        return result



    def insert_new_tone(self, new_region):
        q = f" INSERT INTO tones VALUES (5, '{new_tone}') "
        print('Running: ',q)
        self.cur.execute(q)
        self.con.commit()

    def insert_train_data(self):
        with open("MajorValsNoSquares.txt") as infile:
            for line in infile:
                data = line.split(", ")
                data2 = [x.replace('\n', '') for x in data]
                query = ("insert into tones(tone1,tone2,tone3,tone_key) " "values (%s,%s,%s,'MAJOR')")
                sendtuple = (data2[0], data2[1], data2[2])
                print(sendtuple)
                self.cur.execute(query, sendtuple)
                self.con.commit()
        with open("MinorValsNoSquares.txt") as infile:
            for line in infile:
                data = line.split(", ")
                data2 = [x.replace('\n', '') for x in data]
                query = ("insert into tones(tone1,tone2,tone3,tone_key) " "values (%s,%s,%s,'MINOR')")
                sendtuple = (data2[0], data2[1], data2[2])
                print(sendtuple)
                self.cur.execute(query, sendtuple)
                self.con.commit()

    def mltest(self):

        q='select tone1, tone2, tone3, tone_key from tones'
        df = pd.read_sql(q, self.con)
        df.head()
        df_size = df.size
        df_shape = df.shape
        df_columns = list(df.columns)
        df_targetname = df[df.columns[-1]].name
        df_featurenames = df_columns[0:-1] # select all columns till last column
        df_Xfeatures = df.iloc[:,0:-1]
        df_Ylabels = df[df.columns[-1]]

        # Model Building
        X = df_Xfeatures
        Y = df_Ylabels
        seed = 528
        # prepare models
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))


        # evaluate each model in turn


        train_size = 500

        results = []
        names = []
        allmodels = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            model_selection.p
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            allmodels.append(msg)
            model_results = results
            model_names = names
            print("Name: ")
            print(name)
            print("Message: ")
            print(msg)
            print("Results: ")
            print(cv_results)


        print("Results: ")
        print(*results, sep = "\n")
        print("Models: ")
        print(*allmodels, sep ="\n")

        print(df_targetname)

        return render_template('details.html',
    		df_size=df_size,
    		df_shape=df_shape,
    		df_columns =df_columns,
    		df_targetname =df_targetname,
    		model_results = allmodels,
    		model_names = names,
    		dfplot = df
    		)





@app.route('/')
def index():

    return render_template('index.html')


@app.route('/database')
def database():
    db = Database()
    cl = db.list_tones()
    return render_template('tones.html', result=cl)

@app.route('/test')
def dataupload():
    db = Database()
    db.mltest()
    return render_template('index.html')



# @app.route('/insertTrain')
# def database():
#     db2 = Database()
#     cl = db2.insert_train_data()
#     return '<h1>Inserting Data...</h1>'

if __name__ == '__main__':
    app.run()
