import os
import librosa
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt


# Functie care extrage features (un vector de mfcc uri) dintr-un fisier audio dat
def extract_features(audio_file_name):
    try:
        audio, sample_rate = librosa.load(audio_file_name)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Putem folosi ca features lista de medii ale listelor de mfcc uri, ori le putem inlantui,
        # pentru a obtine o lista unidimensionala (rezultate asemanatoare)

        # mfccsscaled = np.mean(mfccs, axis=1)
        mfccsscaled = [x for sublist in mfccs for x in sublist]

    except Exception as e:
        print("Error with file: ", audio_file_name)
        return None

    return mfccsscaled


# Citim fisierele audio (path ul la care se afla si csv ul cu numele si label urile)
path_train = 'data\\train\\'
path_test = 'data\\test\\'
path_valid = 'data\\validation\\'
train = pd.read_csv('data\\train.txt')
test = pd.read_csv('data\\test.txt')
valid = pd.read_csv('data\\validation.txt')

# Extragem features pt fiecare audio train
features_train = []
for index, row in train.iterrows():
    file_name = os.path.join(path_train, row['name'])
    class_label = row['label']
    data = extract_features(file_name)
    print('Extras mfcc pt ' + str(index) + '/' + str(len(train.name)) + ' fisiere de train')
    features_train.append([data, class_label])

# Convertim in dataframe
featuresdf_train = pd.DataFrame(features_train, columns=['feature', 'class_label'])
print('Terminat extragere pentru ' + str(len(featuresdf_train)) + ' fisiere de train')

# Extragem features pt fiecare audio de validare
features_valid = []
for index, row in valid.iterrows():
    file_name = os.path.join(path_valid, row['name'])
    class_label = row['label']
    data = extract_features(file_name)
    print('Extras mfcc pt ' + str(index) + '/' + str(len(valid.name)) + ' fisiere de validare')
    features_valid.append([data, class_label])

# Convertim in dataframe
featuresdf_valid = pd.DataFrame(features_valid, columns=['feature', 'class_label'])
print('Terminat extragere pentru ' + str(len(featuresdf_valid)) + ' fisiere de validare')

# Extragem features pt fiecare audio test
features_test = []
for index, row in test.iterrows():
    file_name = os.path.join(path_test, row['name'])
    data = extract_features(file_name)
    print('Extras mfcc pt ' + str(index) + '/' + str(len(test.name)) + ' fisiere de test')
    features_test.append([data])

# Convertim in dataframe
featuresdf_test = pd.DataFrame(features_test, columns=['feature'])
print('Terminat extragere pentru ' + str(len(featuresdf_test)) + ' fisiere de test')

# Convertim datele in np.array uri
X_train = np.array(featuresdf_train.feature.tolist())
y_train = np.array(featuresdf_train.class_label.tolist())
X_valid = np.array(featuresdf_valid.feature.tolist())
y_valid = np.array(featuresdf_valid.class_label.tolist())
X_test = np.array(featuresdf_test.feature.tolist())

''' 
# Am folosit partea asta de cod pentru a cauta cel mai bun C pentru classifier
max_score = 0
best_c = 0
for i in range(5, 15):
    print(i)
    model = svm.SVC(kernel='rbf', C=i)
    model.fit(X_train, y_train)
    predicted = model.predict(X_valid)
    score = accuracy_score(y_valid, predicted)
    if max_score < score:
        best_c = i
        max_score = score
print(max_score)
print(best_c)
'''

# Folosim un SVM pentru a clasifica fisierele de test
model = svm.SVC(kernel='rbf', C=9)   # Pentru C = 9 am obtinut cel mai bun scor pe dataset ul de validare

# Antrenam clasificatorul
model.fit(X_train, y_train)
# model.fit(X_valid, y_valid)

'''
# Prezicerea pe validare si afisarea scorului
predicted = model.predict(X_valid)
score = accuracy_score(y_valid, predicted)
print("Accuracy score = " + str(score))
'''

# Obtinem prezicerile pentru set ul de test
predicted = model.predict(X_test)

print(predicted)

# Convertim rezultatele obtinute intr-un dataframe nou si il transformam in csv-ul final
subm = pd.DataFrame()
subm['name'] = test.name
subm['label'] = predicted
subm.to_csv('submission.txt', index=False)

# Scoruri
# f1_scor = f1_score(y_valid, predicted)
# print("f1_score = " + str(f1_scor))

# recall_scor = recall_score(y_valid, predicted)
# print("recall score = " + str(recall_scor))

# precision_scor = precision_score(y_valid, predicted)
# print("precision_score = " + str(precision_scor))

# conf_mat = confusion_matrix(y_valid, predicted)
# print("Confusion matrix: ")
# print(conf_mat)

'''
# Afisarea confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model, X_valid, y_valid,
                                 display_labels=featuresdf_valid.class_label,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()
'''
