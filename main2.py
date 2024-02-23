import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Загрузка датасета Iris
iris = load_iris()
X = iris.data #Массив значений
y = iris.target #Массив правильных ответов

# Преобразование меток классов в бинарный формат
# Было [0,...,1,..,2...], где каждая цифра обозначает вид ириса
# Стало [[1,0,0], [0,1,0], [0,0,1]]
def to_categorical(y):
    y_cat = np.zeros((len(y), 3))
    for i in range(len(y)):
        y_cat[i, y[i]] = 1
    return y_cat

# Ниже два варианта нормализации значений в массиве данных ирисов(дают +- одинаковые значения)

# Нормализация значений
def normalized(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Мин макс нормализация диапазон от 0 до 1
def minMaxNormalized(x):
    return (x - np.min(x, axis=0))/(np.max(x, axis=0) - np.min(x, axis=0))


# Разделение датасета на обучающее и тестовое множество
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


X_norm_train = normalized(X_train)
X_norm_test = normalized(X_test)
y_cat = to_categorical(y_train)

input_dim = X.shape[1] #Кол-во входных данных (4)
hidden_dim = 10 #Кол-во нейронов в скрытом слое
out_dim = 3 #Кол-во выходных классов

#Задаёт рандомные значения для весов и смещения первого входного слоя (Инициализация Ксавье)
#Нужно для нормализации весов, все веса будут инициализироваться в диапазоне от 0 до 1
stddev = np.sqrt(2.0 / (input_dim + out_dim))
weights_in = np.random.normal(0, stddev, (input_dim, hidden_dim))
bios_in = np.random.rand(hidden_dim)

#Задаёт рандомные значения для весов и смещения второго скрытого слоя (Инициализация Ксавье)
weights_hid = np.random.normal(0, stddev, (hidden_dim, out_dim))
bios_hid = np.random.rand(out_dim)

def relu(vector): #функция активации (принимает вектор)
    return np.maximum(vector,0) #Максимум из пришедшего значения и нуля

def relu_derivative(vector): # Производная для функции активации 
    return np.where(vector> 0 , 1, 0)


def softmax(vector): #превращает произвольный вектор в набор вероятностей
    exp_v = np.exp(vector - np.max(vector, axis=1, keepdims=True))
    return exp_v / np.sum(exp_v, axis=1, keepdims=True)


learning_rate = 0.001 #скорость обучения
epochs = 400 #циклы(эпохи)
errors=[]# Массив ошибок для грфика
weights_in_changes =[]# Массив весов для графика

for epoch in range(epochs):
  
    #Вектор первого слоя.
    #Матричное умножение входных параметров и весов на первом слое 
    vector_in = X_norm_train @ weights_in + bios_in
  
    #Активация значений(в виде ветора) первого слоя
    input_ex = relu(vector_in)
    
    #Вектор второго слоя.
    #Матричное умножение входных параметров и весов на втором слое 
    vector_hid = input_ex @ weights_hid + bios_hid
    #Приведение значений скрытого слоя к вероятностям
    exits = softmax(vector_hid)

    #Вычисление кросс-энтропии для построения графика с ошибками
    loss = -np.mean(y_cat * np.log(exits))
    errors.append(loss)
    
    #Обратное распространие 
    #Подсчёт цены ошибки
    error = y_cat - exits
    
    #Подсчёт дельты(пока хз)
    out_del = error * relu_derivative(exits)
    hid_er = out_del @ weights_hid.T
    hid_del = hid_er * relu_derivative(input_ex)

    #Обновление весов и смещений выходной слой
    weights_hid +=input_ex.T @ out_del * learning_rate
    bios_hid += np.sum(out_del, axis=0) * learning_rate
    #Обновление весов и смещений входной слой
    weights_in+= X_norm_train.T @ hid_del * learning_rate
    if(epoch % 200 == 0):
        weights_in_changes.append(weights_in)
    bios_in += np.sum(hid_del, axis=0) * learning_rate

# Тестирование данных
def predict(X):
    vector_in = X @ weights_in + bios_in
    input_ex = relu(vector_in)
    
    vector_hid = input_ex @ weights_hid + bios_hid
    exits = softmax(vector_hid)
    
    return exits
# Построение графика обучения
def graphErr(arrErr, epochs):
    plt.plot(range(epochs), arrErr)
    plt.xlabel('Epochs')
    plt.ylabel('Loss price')
    plt.show()

graphErr(errors, epochs)

# Построение графика изменения входных весов
def graphWeightsChange(weightArr):
    plt.figure(figsize=(10, 5))
    for i, weights in enumerate(weightArr):
        plt.plot(weights, label=f'Layer {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Weights')
    plt.title('Changes in Weights')
    plt.legend()
    plt.show()

graphWeightsChange(weights_in_changes)

# Прогон по тренировочным значениям в среднем 96-98% точности
pred_train = predict(X_norm_train)
accuracy_train = (np.mean(np.argmax(pred_train, axis=1)==y_train))
accuracy_rd_per_train = round(accuracy_train, 4) * 100
print("Точность модели на обучающем наборе данных: ", accuracy_rd_per_train, "%")

# Прогон по тестовым значениям, которые нейросеть ещё не видела в среднем 93-96% точности
pred_test = predict(X_norm_test)
accuracy_test = (np.mean(np.argmax(pred_test, axis=1)==y_test))
accuracy_rd_per_test = round(accuracy_test, 4) * 100
print("Точность модели на тестовом наборе после обучения: ", accuracy_rd_per_test, "%")


