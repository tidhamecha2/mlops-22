# mlops-22
Here is the current output of `python plot_graphs.py`

```
Found new best metric with :{'gamma': 0.01, 'C': 0.1}
New best val metric:0.08839779005524862
Found new best metric with :{'gamma': 0.01, 'C': 0.5}
New best val metric:0.23756906077348067
Found new best metric with :{'gamma': 0.01, 'C': 0.7}
New best val metric:0.4696132596685083
Found new best metric with :{'gamma': 0.01, 'C': 1}
New best val metric:0.8397790055248618
Found new best metric with :{'gamma': 0.01, 'C': 2}
New best val metric:0.856353591160221
Found new best metric with :{'gamma': 0.005, 'C': 0.5}
New best val metric:0.9613259668508287
Found new best metric with :{'gamma': 0.005, 'C': 0.7}
New best val metric:0.9834254143646409
Found new best metric with :{'gamma': 0.001, 'C': 0.7}
New best val metric:0.988950276243094
Found new best metric with :{'gamma': 0.0001, 'C': 5}
New best val metric:0.994475138121547
Classification report for classifier SVC(C=10, gamma=0.0001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        15
           1       0.96      1.00      0.98        22
           2       1.00      1.00      1.00        13
           3       1.00      0.93      0.96        14
           4       1.00      0.89      0.94        18
           5       1.00      0.94      0.97        18
           6       1.00      1.00      1.00        16
           7       1.00      1.00      1.00        15
           8       0.96      0.96      0.96        26
           9       0.88      1.00      0.94        22

    accuracy                           0.97       179
   macro avg       0.98      0.97      0.98       179
weighted avg       0.97      0.97      0.97       179


Best hyperparameters were:
{'gamma': 0.0001, 'C': 5}
```




```
docker build -t exp:v1 -f docker/Dockerfile .
docker run -it exp:v1
```

```
export FLASK_APP=api/app.py ; flask run
```