from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
import numpy as np

def to_dummy_pandas(param) :
  result = list()
  for val in param:
    if val == "RENDAH" :
      result.append([1,0,0])
    elif val == "TINGGI" :
      result.append([0,1,0])
    elif val == "TUA":
      result.append([0,0,1])
    else:
      result.append([0,0,0])
  return [item for sublist in result for item in sublist]

def visualization(hasil_akhir, data_attr_dummy, target_names):
  dot_data = tree.export_graphviz(hasil_akhir,
                                  feature_names=data_attr_dummy.columns,
                                  class_names=target_names,
                                  filled=True, rounded=True)
  graph = graphviz.Source(dot_data)
  graph.render("dec_tree3_dummy", view=True)

def set_error_rate(color, hasil_akhir, target_to_float):
  dataset_attr = color.iloc[:, [0,1,2]]
  error = 0

  for i, record in enumerate(target_to_float):
    if hasil_akhir.predict([to_dummy_pandas(dataset_attr.values[i])])[0] != record:
      error += 1 
  print("Error Rate = {}".format(error/len(target_to_float)))

def user_test(hasil_akhir):
  user_input = input('Masukkan nilai atribut (RED,GREEN,BLUE) : ').split(',')
  print('Hasil: ')
  print('BACKGROUND' if hasil_akhir.predict([to_dummy_pandas(user_input)])[0] == 0 else 'OBJEK')

def hitung():
  color = pd.read_csv('warna.csv')

  dummy_color = pd.get_dummies(color)
  data_attr_dummy = dummy_color.iloc[:, [0,1,2,3,4,5,6,7,8]]
  data_value_dummy = data_attr_dummy.values

  target_color = color.iloc[:, 3].values
  target_to_float = np.array(target_color)
  target_names = np.unique(target_to_float)
  target_to_float = np.where(target_to_float=='OBYEK', 1, 0)

  clf = tree.DecisionTreeClassifier()
  hasil_akhir = clf.fit(data_value_dummy, target_to_float)

  set_error_rate(color, hasil_akhir, target_to_float)
  visualization(hasil_akhir, data_attr_dummy, target_names)
  user_test(hasil_akhir)

def main():
  hitung()

if __name__ == "__main__":
  main()