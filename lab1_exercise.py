# KNN (k nearest neighbours) - K najbliższych sąsiadów

# Jeden z algorytmów uczenia nadzorowanego służący do klasyfikacji i regresji.
# Zasada działania algorytmu w zadaniu klasyfikacji polega na przydzieleniu klasy decyzyjnej,
# którą obiekty w określonym zasięgu najliczniej reprezentują.
# W zadaniu regresji predykcja przydzielana jest na podstawie uśrednionej predykcji obiektów znajdujących
# się w określonym zasięgu. Zasięgiem w przypadku algorytmu KNN jest liczba najbliższych sąsiadów,
# czyli obiektów pochodzących z systemu decyzyjnego umieszczonych w przestrzeni n-wymiarowej.

import random
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Przykład - przewidywanie cen domów

# Wczytanie systemu decyzyjnego i wstępna eksploracja

houses = pd.read_csv('./data/data.csv')

# print(houses.head(3))

# W obiekcie values umieszczamy atrybut decyzyjny z oryginalnego systemu decyzyjnego,
# następnie usuwamy atrybut z systemu decyzyjnego uzyskując w ten sposób system informacyjny w obiekcie houses.

values = houses['AppraisedValue']
houses.drop('AppraisedValue', 1, inplace=True)

# Dokonujemy normalizacji danych w systemie informacyjnym
# (zerowa średnia arytmetyczna i jednostkowe odchylenie standardowe),
# oraz usuwamy zbędne atrybuty pozostawiając jedynie współrzędne geograficzne i wielkość działki.
# Na podstawie tych atrybutów będziemy przewidywać docelowe ceny domów.

houses = (houses - houses.mean()) / (houses.max() - houses.min())
houses = houses[['lat', 'long', 'SqFtLot']]

# Tworzymy obiekt regressora KNN

kdtree = KDTree(houses)

# Następnie tworzymy funkcję, która przydzieli prognozę przekazanemu obiektowi
# na podstawie liczby najbliższych sąsiadów (parametr k).


def predict(query_point, k):
    _, idx = kdtree.query(query_point, k)
    return np.mean(values.iloc[idx])

# Tworzymy następnie systemy decyzyjne służące do trenowania i testowania modelu.
# System treningowy będzie zawierał 80% obiektów z oryginalnego systemu decyzyjnego, a system testowy 20%.


test_rows = random.sample(houses.index.tolist(), int(round(len(houses) * .2)))  # 20%
train_rows = set(range(len(houses))) - set(test_rows)
df_test = houses.loc[test_rows]
df_train = houses.drop(test_rows)
test_values = values.loc[test_rows]
train_values = values.loc[train_rows]


# Dla wartości K (liczba najbliższych sąsiadów) równej 5 dokonujemy regresji obiektów pochodzących z systemu testowego,
# a następnie porównujemy wyniki z wynikami oryginalnymi przy użyciu funkcji średniego błędu bezwzględnego.


#train_predicted_values = []
#train_actual_values = []

#for _id, row in df_train.iterrows():
    #train_predicted_values.append(predict(row, 5))
    #train_actual_values.append(train_values[_id])


#print(f'Wartosc sredniego bledu bezwzglednego na systemie treningowym dla k=5 wynosi: '
      #f'{mean_absolute_error(train_predicted_values, train_actual_values)}')



# Zadanie


# 1. Wykorzystać powyższy przykład w celu znalezienia takiej wartości K, dla której wartość średniego błędu b
# ezwzględnego na systemie treningowym będzie najmniejsza. W tym celu należy stworzyć wykres liniowy,
# na którym oś x będzie przedstawiała wartość K, a oś y będzie przedstawiała wartość funkcji błędu.
# Następnie dla "najlepszej" wartości K sprawdzić wartość funkcji błędu na systemie testowym.

# 2. Znaleźć "najlepszą" wartość K dla następujących podziałów na system treningowy i testowy:
# 60% system treningowy i 40% system testowy
# 65% system treningowy i 35% system testowy
# 70% system treningowy i 30% system testowy
# 75% system treningowy i 25% system testowy


# Zad. 1

error_values = []
max_k = 7

for i in range(1, max_k):
    train_predicted_values = []
    train_actual_values = []

    for _id, row in df_train.iterrows():
        train_predicted_values.append(predict(row, i))
        train_actual_values.append(train_values[_id])

    error_values.append(mean_absolute_error(train_predicted_values, train_actual_values))

print(error_values)
# [5.436856161770317, 43161.33077785197, 52049.28989412438, 56557.50497782335, 59288.07842426555, 61366.96037612871]

plt.plot(range(1, max_k), error_values)
plt.show()
