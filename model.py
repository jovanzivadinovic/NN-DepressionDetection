import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Data preproccessing

df = pd.read_csv('data.csv')

df_clean = df.dropna()

label_encoder = LabelEncoder()


df_clean['gender'] = label_encoder.fit_transform(df_clean['gender'])
df_clean['who_bmi'] = label_encoder.fit_transform(df_clean['who_bmi'])
df_clean['depression_severity'] = label_encoder.fit_transform(df_clean['depression_severity'])
df_clean['depressiveness'] = label_encoder.fit_transform(df_clean['depressiveness'])
df_clean['suicidal'] = label_encoder.fit_transform(df_clean['suicidal'])
df_clean['depression_diagnosis'] = label_encoder.fit_transform(df_clean['depression_diagnosis'])
df_clean['depression_treatment'] = label_encoder.fit_transform(df_clean['depression_treatment'])
df_clean['anxiety_severity'] = label_encoder.fit_transform(df_clean['anxiety_severity'])
df_clean['anxiousness'] = label_encoder.fit_transform(df_clean['anxiousness'])
df_clean['anxiety_diagnosis'] = label_encoder.fit_transform(df_clean['anxiety_diagnosis'])
df_clean['anxiety_treatment'] = label_encoder.fit_transform(df_clean['anxiety_treatment'])
df_clean['sleepiness'] = label_encoder.fit_transform(df_clean['sleepiness'])

del df_clean['id']

X = df_clean.drop('depression_diagnosis',axis=1)
y = df_clean.pop('depression_diagnosis')



ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

# Creating a model

model = Sequential()

model.add(Dense(17, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience=25)

model.fit(x=X_train, y=y_train, epochs=140, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stop], batch_size=32)

# Evaluating the model

train_accuracy = model.history.history['accuracy']

val_accuracy = model.history.history['val_accuracy']

train_loss = model.history.history['loss']

val_loss = model.history.history['val_loss']

model.evaluate(X_test, y_test)

test_predictions = model.predict(X_test)

bin_predictions = (test_predictions >= 0.5).astype(int)


plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')
plt.title('LOSS')
plt.legend()
plt.show()

plt.plot(train_accuracy, label='train_accuracy')
plt.plot(val_accuracy, label='val_accuracy')
plt.title('ACCURACY')
plt.legend()
plt.show()




print(classification_report(y_test, bin_predictions))

print(confusion_matrix (y_test, bin_predictions))