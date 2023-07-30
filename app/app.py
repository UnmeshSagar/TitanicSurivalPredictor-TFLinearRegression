import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

#Loading dataset
df_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
df_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = df_train.pop('survived')
y_eval = df_eval.pop('survived')


#input function
def makeInputFn(dataDf, labelDf, numEpochs=10, shuffle=True, batch_size=32):
  def inputFunction():
    ds = tf.data.Dataset.from_tensor_slices((dict(dataDf), labelDf))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(numEpochs)
    return ds
  return inputFunction
trainInputFn = makeInputFn(df_train, y_train)
evalInputFn = makeInputFn(df_eval, y_eval, numEpochs=1, shuffle=False)


categoricalColumns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
numericalColumns = ['age', 'fare']

feature_columns = []
for featureName in categoricalColumns:
  vocabulary = df_train[featureName].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(featureName, vocabulary))

for featureName in numericalColumns:
  feature_columns.append(tf.feature_column.numeric_column(featureName, dtype=tf.float32))

trainInputFunc = makeInputFn(df_train, y_train, numEpochs=20)
evalInputFunc = makeInputFn(df_eval, y_eval, numEpochs=1, shuffle=False)

linearEstimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)

#train
linearEstimator.train(trainInputFn)

#testing 
result = list(linearEstimator.predict(evalInputFunc))
choice=int(input('Enter your Choice:\n1.Predict a persons survival chance:\n2.Plot the graph of survial:\n \t: '))

#output
match(choice):
  case 1:
    getNum = int(input('Enter the Person number: '))
    print('Details of Person: \n',df_eval.loc[getNum])
    print(f"Probability of Survival is : {result[getNum]['probabilities'][1]}")
    print(f'Actual possibility of survival is: {y_eval.loc[getNum]}')
  case 2:
    probs = pd.Series([pred['probabilities'][1] for pred in result])
    getNum=int(input('Enter number of data to considered while Plotting: '))
    ax = probs.plot(kind='hist', bins=getNum, title='Predicted Probabilities')
    ax.set_xlabel('Probability in %')
    ax.set_ylabel('Frequency')
  case _:
    print('Wrong input')