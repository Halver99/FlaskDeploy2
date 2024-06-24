from flask import Flask, render_template, request, redirect, session, url_for
import csv
import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
#nltk.download('stopwords')
#nltk.download('punkt')
import matplotlib.pyplot as plt
from collections import Counter
from flask import send_from_directory
#Pembobotan TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#hitung
from wordcloud import WordCloud

#SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import svm


import math
app = Flask(__name__)
app.secret_key = 'harrymantap1'

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
    
ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER'] = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/uploaddata', methods=['GET', 'POST'])
def uploaddata():
    if request.method == 'GET':
        return render_template('uploaddata.html')
    
    elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file.filename = "dataset.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Validate CSV columns
            text = pd.read_csv(filepath, encoding='latin-1')
            if 'content' not in text.columns or 'Label' not in text.columns:
                flash('Data tidak valid. File harus memiliki dua kolom dengan nama "content" dan "Label".')
                return redirect(request.url)
            
            total_data = text.shape[0]  # Menghitung jumlah baris dalam dataframe text
            training = total_data * (70/100)
            testing = total_data * (30/100)
            # membulatkan ke bawah menggunakan math.floor
            training_data_floor = math.floor(training)  # Hasilnya akan menjadi 3364
            testing_data_floor = math.floor(testing)  # Hasilnya akan menjadi 1441
            
            positif, negatif = hitung()
            # Karena total data harus tetap 4806, kita tambahkan sisa pembulatan ke data testing
            testing_data_floor2 = testing_data_floor + (total_data - (training_data_floor + testing_data_floor))  
            return render_template('uploaddata.html', tables=[text.to_html()], total_data=total_data, data_uji = testing_data_floor2, data_latih = training_data_floor, data_positif = positif, data_negatif = negatif)

def hitung():
    data = pd.read_csv('uploads/dataset.csv',  encoding='latin-1')
    jumlah = data['Label'].value_counts()

    positif = jumlah['Positif']
    negatif = jumlah['Negatif']

    return positif, negatif
    


@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    
    if request.method == 'GET':
        return render_template('klasifikasi.html', df_test_subset=pd.DataFrame())
    
    elif request.method == 'POST':
        ulasan = preprocess_ulasan()
        accuracy, precision, recall, f1, tp, fn, fp, tn, df_test, df_test_subset = testing()
       
        session['results'] = accuracy, precision.tolist(), recall.tolist(), f1.tolist(), tp.item(), fn.item(), fp.item(), tn.item()


        # Mengembalikan hasil ke template HTML
        return render_template('klasifikasi.html', df_test_subset = df_test_subset)



@app.route('/status')
def status():
    if 'results' in session:
        accuracy, precision, recall, f1, tp, fn, fp, tn = session['results']

        return 'selesai'
    else : 
        return 'processing'


def preprocess_ulasan():
    global ulasan
    ulasan = pd.read_csv('uploads/dataset.csv', encoding='latin-1')
    def cleaningulasan(ulasan):
        ulasan = re.sub(r'@[A-Za-a0-9]+',' ',ulasan)
        ulasan = re.sub(r'#[A-Za-z0-9]+',' ',ulasan)
        ulasan = re.sub(r"http\S+",' ',ulasan)
        ulasan = re.sub(r'[0-9]+',' ',ulasan)
        ulasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", ulasan)
        ulasan = ulasan.strip(' ')
        return ulasan
    ulasan['cleaning']= ulasan['content'].apply(cleaningulasan)

    def clearEmoji(ulasan):
        return ulasan.encode('ascii', 'ignore').decode('ascii')
    ulasan['hapusEmoji']= ulasan['cleaning'].apply(clearEmoji)
    def replaceTOM(ulasan):
        pola = re.compile(r'(.)\1{2,}', re.DOTALL)
        return pola.sub(r'\1', ulasan)
    ulasan['replaceTOM']= ulasan['hapusEmoji'].apply(replaceTOM)
    def casefoldingText(ulasan):
        ulasan = ulasan.lower()
        return ulasan
    ulasan['caseFolding']= ulasan['replaceTOM'].apply(casefoldingText)
    #TOKENIZING
    def tokenizingText(ulasan):
        ulasan = word_tokenize(ulasan)
        return ulasan

    ulasan['tokenizing']= ulasan['caseFolding'].apply(tokenizingText)
    #formalisasi kata 

    def convertToSlangword(ulasan):
        kamusSlang = eval(open("uploads/slangwords.txt").read())
        pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
        content = []
        for kata in ulasan:
            filterSlang = pattern.sub(lambda x: kamusSlang[x.group()],kata)
            content.append(filterSlang.lower())
        ulasan = content
        return ulasan
    ulasan['formalisasi'] = ulasan['tokenizing'].apply(convertToSlangword)
    #stopword
    daftar_stopword = stopwords.words('indonesian')
    # ---------------------------- manualy add stopword  ------------------------------------
    # append additional stopword
    daftar_stopword.extend(["yg","dg","rt","nya","eh","yah","kan","ke","di", "ter"])
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        print(words)
        return [word for word in words if word not in daftar_stopword]

    ulasan['stopwordRemoval'] = ulasan['formalisasi'].apply(stopwordText)
    #stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in ulasan['stopwordRemoval']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])

    def stemmingText(document):
        return [term_dict[term] for term in document]
    
    ulasan['stemming'] = ulasan['stopwordRemoval'].swifter.apply(stemmingText)
    
    return ulasan


def testing():
    X = ulasan['stemming']
    Y = ulasan['Label']

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=64)

    # Simpan teks asli x_test ke dalam variabel lain
    x_test_text = x_test.copy()

    nnn = x_test[1090]
    #print(nnn)

    def dummy_fun(doc):
        return doc
    vectorizer = TfidfVectorizer(analyzer='word',
        norm ='l2',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(y_train)
    y_test = Encoder.fit_transform(y_test)
    
    # Linear kernel with cost=1
    linear_SVM = svm.SVC(kernel='linear', C=1, probability=True)  # Cost parameter added
    linear_SVM.fit(x_train, y_train)

    y_pred_linear = linear_SVM.predict(x_test)
    # acurracy
    accuracy = linear_SVM.score(x_test, y_test)
    # Calculate precision
    precision = precision_score(y_test, y_pred_linear, average='weighted')
    # Calculate recall
    recall = recall_score(y_test, y_pred_linear, average='weighted')
    # Calculate F1-score
    f1 = f1_score(y_test, y_pred_linear, average='weighted')
    cm = confusion_matrix(y_test, y_pred_linear)
    tp, fn, fp, tn = cm.ravel()
    #Print accuracy results
    #print('Hasil Rata - Rata Accuracy (Linear Kernel):', acc_score_linear.mean())

    # Create a DataFrame for the test data
    df_test = pd.DataFrame({
        'stemming': x_test_text,
        'Hasil sebenarnya': y_test,
        'Prediksi': y_pred_linear
    })
    #mengubah kebentuk teks kembali
    df_test['teks'] = df_test['stemming'].apply(lambda tokens: ' '.join(tokens))
    # Mengubah nilai 0 dan 1 pada kolom 'Hasil sebenarnya' dan 'Prediksi'
    df_test['Hasil sebenarnya'] = df_test['Hasil sebenarnya'].replace({0: 'Negatif', 1: 'Positif'})
    df_test['Prediksi'] = df_test['Prediksi'].replace({0: 'Negatif', 1: 'Positif'})

    df_test_subset = df_test [['teks' , 'Hasil sebenarnya', 'Prediksi']]

    # Simpan df_test ke dalam folder "uploads"
    app.config['UPLOAD_FOLDER'] = 'uploads'
    df_test.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'df_test.csv'), index=True)
    return  accuracy, precision, recall, f1, tp, fn, fp, tn, df_test, df_test_subset


@app.route('/visualisasi', methods=['GET', 'POST'])
def visualisasi():
    if request.method == 'GET':
        # Load df_test from the CSV file
        df_test_path = os.path.join('uploads', 'df_test.csv')
        if os.path.exists(df_test_path):
            df_test = pd.read_csv('uploads/df_test.csv',  encoding='latin-1')
            # Pisahkan teks positif dan negatif berdasarkan polarity
            positive_text_data = df_test[df_test['Hasil sebenarnya'] == 'Positif']['teks'].dropna().str.cat(sep=' ')
            negative_text_data = df_test[df_test['Hasil sebenarnya'] == 'Negatif']['teks'].dropna().str.cat(sep=' ')

            # Buat word cloud untuk nilai positif
            wordcloud_positive = WordCloud(width=800, height=400, background_color='black').generate(positive_text_data)

            # Tampilkan word cloud untuk nilai positif
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_positive, interpolation='bilinear')
            plt.title('Positive Word Cloud')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('static/positif.png')

            # Buat word cloud untuk nilai negatif
            wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(negative_text_data)

            # Tampilkan word cloud untuk nilai negatif
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_negative, interpolation='bilinear')
            plt.title('Negative Word Cloud')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('static/negatif.png')
            
            # Tokenize teks
            positive_tokens = word_tokenize(positive_text_data)
            negative_tokens = word_tokenize(negative_text_data)

            # Hitung frekuensi kata
            positive_word_freq = Counter(positive_tokens)
            negative_word_freq = Counter(negative_tokens)

            # Pilih 20 kata dengan frekuensi tertinggi
            positive_top_words = positive_word_freq.most_common(20)
            negative_top_words = negative_word_freq.most_common(20)

            # Visualisasikan frekuensi kata
            def plot_word_freq(word_freq, title, color):
                words = [item[0] for item in word_freq]
                freqs = [item[1] for item in word_freq]

                plt.figure(figsize=(10, 5))
                bars = plt.barh(words, freqs, color=color)
                plt.xlabel('Frekuensi')
                plt.ylabel('Kata')
                plt.title(title)
                plt.gca().invert_yaxis()

                for bar in bars:
                    width = bar.get_width()
                    plt.text(width, bar.get_y() + bar.get_height()/2, f' {width}', va='center')
            # Plot frekuensi kata positif
            plot_word_freq(positive_top_words, 'Frekuensi Kata Positif', 'green')
            plt.tight_layout()
            plt.savefig('static/frekuensi_positif.png')

            # Plot frekuensi kata negatif
            plot_word_freq(negative_top_words, 'Frekuensi Kata Negatif', 'red')
            plt.tight_layout()
            plt.savefig('static/frekuensi_negatif.png')
            

            # Generate word cloud for negative 
            negative_wordcloud_path = url_for('static', filename='negatif.png')
            # Generate word cloud for positive t
            positive_wordcloud_path = url_for('static', filename='positif.png') 
            # Generate word cloud for negative 
            negative_freq_path = url_for('static', filename='frekuensi_negatif.png') 
            # Generate word cloud for positive 
            positive_freq_path = url_for('static', filename='frekuensi_positif.png') 
            
            return render_template('visualisasi.html', negative_wordcloud=negative_wordcloud_path, positive_wordcloud=positive_wordcloud_path, positive_freq=positive_freq_path, negative_freq=negative_freq_path)
        else:
            return render_template('visualisasi.html')


@app.route('/klasifikasisvm', methods=['GET', 'POST'])
def klasifikasisvm():
    if request.method == 'GET':
        
        # Check if results are available
        if 'results' in session:
            accuracy, precision, recall, f1, tp, fn, fp, tn = session['results']
            df_test = pd.read_csv('uploads/df_test.csv',  index_col=0)
            df_test_subset = df_test[['teks', 'Hasil sebenarnya', 'Prediksi']]

            return render_template(
                'klasifikasisvm.html', 
                accuracy_score_linear=accuracy, 
                precision_score_linear=np.array(precision), 
                recall_score_linear=np.array(recall), 
                f1_score_linear=np.array(f1),
                tp_linear=np.int64(tp),
                fn_linear=np.int64(fn),
                fp_linear=np.int64(fp),
                tn_linear=np.int64(tn),
                df_test_subset=df_test_subset
            )

        else : 
            return render_template('klasifikasisvm.html', df_test_subset=pd.DataFrame())
    
@app.route('/download/<filename>')
def download_file(filename):
    app.config['UPLOAD_FOLDER'] = 'uploads'
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)   

@app.route('/reset', methods=['POST'])
def reset():
    # Remove 'results' from the session
    session.pop('results', None)

    # List of files to delete
    files_to_delete = [
        'static/positif.png',
        'static/negatif.png',
        'static/frekuensi_positif.png',
        'static/frekuensi_negatif.png',
        'uploads/df_test.csv'
    ]

    # Delete files if they exist
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)

    # Redirect to the klasifikasisvm page
    return redirect(url_for('klasifikasisvm'))


if __name__ == '__main__':
    app.run(debug=True)
