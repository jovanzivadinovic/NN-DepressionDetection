import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('data.csv')

X=df[['school_year','age','gender','bmi','who_bmi','phq_score','depression_severity','depressiveness','suicidal','depression_treatment','gad_score','anxiety_severity','anxiousness','anxiety_diagnosis','anxiety_treatment','epworth_score','sleepiness']].values

y = df[['depression_diagnosis']].values

ros = RandomOverSampler(random_state=42)

X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

#########################################

model = Sequential()

model.add(Dense(17, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience=25)

model.fit(x=X_train, y=y_train, epochs=350, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stop], batch_size=16)

######################################################################

train_accuracy = model.history.history['accuracy']

val_accuracy = model.history.history['val_accuracy']

train_loss = model.history.history['loss']

val_loss = model.history.history['val_loss']

model.evaluate(X_test, y_test)

test_predictions = model.predict(X_test)

bin_predictions = (test_predictions >= 0.5).astype(int)

print(classification_report(y_test, bin_predictions))

print(confusion_matrix (y_test, bin_predictions))