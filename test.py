import os
import pickle
from sklearn.externals import joblib

dir_path = '/home/silence/proj/'

labels = [1, 0, 0, 1,
 0, 1, 1, 0, 
1, 1, 1, 1,
 0, 0, 1, 2,
 1, 1, 1, 1,
 1, 1, 1, 2,
 1, 1, 1, 1, 
1, 1, 2, 2,
1, 0, 0, 1,
2, 1, 1, 1,
1, 2, 1, 2,
1, 1, 2, 0,
1, 2, 1, 2,
1, 0, 1, 2,
1, 2, 2, 1,
1, 1, 1, 1,
1, 1, 1, 1,
1, 1, 0, 1,
1, 1, 1, 1,
1, 1, 2, 2,
2, 0, 2, 0,
2, 0, 0, 1,
1, 1, 1, 0,
2, 1, 1, 1,
1, 1, 1, 2,
2, 1, 0, 2,
2, 2, 2, 1,
2, 2, 0, 2,
2, 2, 1, 1,
1, 1, 1, 1,
1, 1, 2, 1,
1, 1, 1, 1,
1, 2, 1, 1,
2
]
labels2 = []

for i in range(len(labels)):
    if labels[i] == 0:
        labels2.append(1)
        
    elif labels[i]==1:
        labels2.append(2)
        
    else:
        labels2.append(0)
  

features = joblib.load(dir_path+'clusters.pkl')
correct = 0
for i in range(len(features)):
    if features[i] == labels2[i]:
        correct += 1
print(correct/len(features))
print(features)

