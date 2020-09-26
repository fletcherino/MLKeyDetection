from flask import Flask, render_template,request
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
import pymysql
import credentials
import pandas as pd
import matplotlib.pyplot as plt

from werkzeug.utils import secure_filename


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
from sklearn import svm


# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
bootstrap = Bootstrap(app)

files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)



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



    def insert_new_tone(self, freqins):
        if freqins[0] is not "" and freqins[1] is not "" and freqins[2] is not "":
            query = (" INSERT INTO tones(tone1,tone2,tone3,tone_key) " "VALUES (%s,%s,%s,%s) ")
            sendtuple = (1, float(freqins[1])/float(freqins[0]), float(freqins[2])/float(freqins[0]), freqins[3])
            self.cur.execute(query, sendtuple)
            self.con.commit()
            return render_template('index_insert.html')
        else:
            return render_template('index_error.html')

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



    def mltest(self, freqs):
        if freqs[0] is not "" and freqs[1] is not "" and freqs[2] is not "":
            print(freqs)
            q='select tone1, tone2, tone3, tone_key from tones'
            df = pd.read_sql(q, self.con)
            df.head()
            df_size = df.size
            df_shape = df.shape
            df_columns = list(df.columns)
            df_targetname = df[df.columns[-1]].name
            df_featurenames = df_columns[1:-1] # select all columns till last column
            df_Xfeatures = df.iloc[:,1:-1]
            df_Ylabels = df[df.columns[-1]]

            # Model Building
            X = df_Xfeatures
            Y = df_Ylabels
            seed = 528
            # prepare models
            models = []
            models.append(('K-Neighbours', KNeighborsClassifier()))
            models.append(('Classification and Regression Tree', DecisionTreeClassifier()))
            models.append(('Gaussian Naive Bayes', GaussianNB()))
            models.append(('SVC', svm.SVC()))
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1, random_state=42)
            # clf = svm.SVC()
            # clf.fit(X, Y)

            # print(X_test)






            # evaluate each model in turn


            train_size = 500

            results = []
            names = []
            allmodels = []
            scoring = 'accuracy'
            for name, model in models:
                print(name + " Prediction: ")
                model.fit(X,Y)
                # print(model.predict([[1.261802575107296, 1.497854077253219]])) ##Major
                # print(model.predict([[1.1888412017167382, 1.497854077253219]]))  ##Minor
                # print(model.predict([[1.6823104693140793, 10.086642599277978]])) ##Minor
                # print(model.predict([[2.3776824034334765, 5.9957081545064375]])) ##Minor
                # print(model.predict([[1.259090909090909, 11.986363636363636]])) ##Major
                # print(model.predict([[1.5884476534296028, 4.759927797833935]])) ##Major
                # print(model.predict([[1.1888412017167382, 2.0]])) ##Minor
                user_in_processed = float(freqs[1])/float(freqs[0]), float(freqs[2])/float(freqs[0])

                predictions = (model.predict([[float(freqs[1])/float(freqs[0]), float(freqs[2])/float(freqs[0])]]))
                predict_text = [i.strip('[]') for i in predictions]
                ratios = float(freqs[0])/float(freqs[0]), float(freqs[1])/float(freqs[0]), float(freqs[2])/float(freqs[0])
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f) %s %s" % (name, cv_results.mean(), cv_results.std(), predict_text, ratios)
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

        else:
            return render_template('index_error.html')





    @app.route('/',methods=['GET','POST'])
    def index():

        return render_template('index.html')

    #
    # @app.route('/database')
    # def database():
    #     db = Database()
    #     cl = db.list_tones()
    #     return render_template('tones.html', result=cl)
    #
    # @app.route('/test')
    # def dataupload():
    #     db = Database()
    #     db.mltest()
    #     return render_template('index.html')
    #
    #
    # @app.route('/insertTrain')
    # def insertTraining():
    #     db = Database()
    #     cl = db.insert_train_data()
    #     return '<h1>Inserting Data...</h1>'

    @app.route('/insert',methods=['GET','POST'])
    def insert_tone():

        db = Database()
        freqins = []
        freq1ins = request.form['freq1ins']
        freq2ins = request.form['freq2ins']
        freq3ins = request.form['freq3ins']
        key_mode = request.form['mode']
        #     file = request.files['csv_data']
        #
        freqins.append(freq1ins)
        freqins.append(freq2ins)
        freqins.append(freq3ins)
        freqins.append(key_mode)

        if request.method == 'POST' and freq1ins is not None and freq2ins is not None and freq3ins is not None:
            return db.insert_new_tone(freqins)

        else:
            return render_template('index_error.html')


    @app.route('/dataupload',methods=['GET','POST'])
    def dataupload1():

        db = Database()
        freqs = []
        freq1 = request.form['freq1']
        freq2 = request.form['freq2']
        freq3 = request.form['freq3']

        freqs.append(freq1)
        freqs.append(freq2)
        freqs.append(freq3)



        print(freq1, freq2, freq3, request.method)

        if request.method == 'POST' and freq1 is not None and freq2 is not None and freq3 is not None:

        #     print("FREQUENCIES PULLED: ")
        #     filename = secure_filename(file.filename)
        #     # os.path.join is used so that paths work in every operating system
        #     # file.save(os.path.join("wherever","you","want",filename))
        #     file.save(os.path.join('static/uploadsDB',filename))
        #     fullfile = os.path.join('static/uploadsDB',filename)
        #
        #     db.mltest(user_input)
        #
        #     print(db.mltest())
        #
        #
        #     # Saving Results of Uploaded Files  to Sqlite DB
        #     newfile = FileContents(name=file.filename,data=file.read(),modeldata=msg)
        #     db.session.add(newfile)
        #     db.session.commit()

            return db.mltest(freqs)


        else:
            return render_template('index_error.html')

        @app.route('/insert-form',methods=['GET','POST'])
        def insert_form():
            db = Database()
if __name__ == '__main__':
    app.run()
