import os
import re
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from flask import Flask, render_template, request, redirect, url_for, abort, session
from werkzeug.utils import secure_filename
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory, ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover

###=>>FLASK ROOT SETTING
project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')

#mendefiniksikan variabel untuk lokasi file
dataset_path = f'{project_root}/dataset/'
stopword_path = f'{project_root}/wordtext/stopwords.txt'
keynorm = f'{project_root}/wordtext/keynorm.csv'
model_path = f'{project_root}/models/'
target_kelas = ['SMS Normal', 'SMS Fraud atau Penipuan', 'SMS Promo']

#mendekrasikan app flask
app = Flask(__name__, template_folder=template_path, static_folder=static_path)
app.secret_key = 'ini kunci rahasia'

#memuat file stopword
with open(stopword_path) as f:
  more_stopword=f.read().split('\n')

#memuat file keynorm untuk normalize text
normalize=dict()
data = pd.read_csv(keynorm)
for singkat, hasil in zip(data['singkat'], data['hasil']):
  normalize[singkat] = hasil

#mendeklarasikan objek untuk proses stopword dan stemmer
SWfactory = StopWordRemoverFactory()
stopword_data = ArrayDictionary(more_stopword+SWfactory.get_stop_words())
stopword = StopWordRemover(stopword_data)
Sfactory = StemmerFactory()
stemmer = Sfactory.create_stemmer()

#mendeklarasikan fungsi preprocessing
def preprocessing(text):
  cleaning = re.sub("http[^*\s]+|<[^>]*>|#[^\W]+|@[^\W]+|[0-9]|[\W]+", ' ', text)#cleaning
  case_folding = cleaning.lower() #case folding
  tokenizing = case_folding.split() #tokenizing
  fix_word = []
  for word in tokenizing:
    normalization=normalize[word] if word in normalize.keys() else word #normalization
    stopword_removal = stopword.remove(normalization) #stopword
    stemming = stemmer.stem(stopword_removal) #stemming
    if stemming != "":
      fix_word.append(stemming)
  return fix_word

#fungsi untuk indentifikasi tokenize
def identity_tokenizer(text):
    return text

#reuter untuk index (halaman pengujian)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        #ambil text dari textbox
        smstext = request.form['smstext']
        #panggi saved model
        tfidf = pickle.load(open(f'{model_path}/tfidf.pkl','rb'))
        NaiveBayes = pickle.load(open(f'{model_path}/model_nb.pkl','rb'))
        #lakukan preprocessing
        sms = preprocessing(smstext)
        #lakukan prediksi dengan model naivebayes
        predict = NaiveBayes.predict(tfidf.transform([sms]).toarray())
        #panggi berdasarkan kelas hasil prediksi
        pred = target_kelas[predict[-1]]
        #kirimkan hasil ke halaman index (pengujian)
        return render_template('index.html', pred_status=True, predict=pred, sms=smstext)
    #tampilkan halaman view index (pengujian)
    return render_template('index.html', pred_status=False)

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == "POST":
        #ambil dataset dari inputan
        f = request.files['dataset']
        #impan dataset ke folder dataset
        csv_dataset = secure_filename(f.filename)
        f.save(dataset_path + csv_dataset)
        
        #panggil dataset
        data = pd.read_csv(dataset_path + csv_dataset)
        X = data['Teks'].values
        y = data['label'].values

        #melakukan preprocessing untuk setiap kalimat
        tokenized_list_of_sentences=[]
        for sentence in tqdm(X):
            preproces=preprocessing(sentence)
            tokenized_list_of_sentences.append(preproces)

        #melakukan feature extraction dengan tfidf
        tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)    
        X = tfidf.fit_transform(tokenized_list_of_sentences).toarray()

        #split dataset menjadi training(80%) dan testing(20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        #latih model naive bayes
        # NaiveBayes = GaussianNB()
        NaiveBayes = MultinomialNB()
        NaiveBayes.fit(X_train, y_train)
        y_pred = NaiveBayes.predict(X_test)
        # Saving model to disk
        os.makedirs(os.path.dirname(model_path), exist_ok=True) 
        pickle.dump(tfidf, open(f'{model_path}tfidf.pkl','wb'))
        pickle.dump(NaiveBayes, open(f'{model_path}model_nb.pkl','wb'))
        #valuasi
        clf_report=classification_report(y_test, y_pred, target_names=target_kelas, output_dict=True)
        pickle.dump(clf_report, open("temp/report.pkl", "wb"))
        #kirimkan ke view pelatihan
        return render_template('training.html', report=clf_report)
    #tampilkan halaman view pelatihan
    return render_template('training.html', report=pickle.load(open("temp/report.pkl", "rb")))

#run flask server
if __name__ == '__main__':
  app.run(host='0.0.0.0',port=80, debug=True)