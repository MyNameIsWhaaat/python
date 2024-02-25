import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    errors=[]# Массив ошибок для графика
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate, epochs):
        #Инициализация скорости обучения и эпох на базе
        self.learning_rate = learning_rate
        self.epochs = epochs

        #Инициализация весов на базе(пока для двух слоёв)
        self.weights_input_hidden = np.random.normal(0, np.sqrt(2.0 / (input_size + output_size)), (input_size, hidden_size))
        self.bios_input_hidden = np.random.rand(hidden_size)

        self.weights_hidden_output = np.random.normal(0, np.sqrt(2.0 / (input_size + output_size)), (hidden_size, output_size))
        self.bios_hidden_output = np.random.rand(output_size)

    #Функция нормализации значений на базе
    def normalized(self, x):
        return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    #Функция активация и производная для функции активации на базе (релу)
    def relu(self,vector):
        return np.maximum(vector,0)

    def relu_derivative(self,vector):
        return np.where(vector> 0 , 1, 0)
        
    #Функция активации для последнего слоя на базе
    def softmax(self, vector):
        exp_v = np.exp(vector - np.max(vector, axis=1, keepdims=True))
        return exp_v / np.sum(exp_v, axis=1, keepdims=True)
    
    #Функция обучения на базе
    def learing(self, X_normalized_train, y_cat):

        for epoch in range(self.epochs):
            
            vector_in = self.normalized(X_normalized_train) @ self.weights_input_hidden + self.bios_input_hidden
            input_ex = self.relu(vector_in)
            
            vector_hid = input_ex @ self.weights_hidden_output + self.bios_hidden_output
            exits = self.softmax(vector_hid)

            loss = -np.mean(y_cat * np.log(exits))
            self.errors.append(loss)
            
            error = y_cat - exits
            
            out_del = error * self.relu_derivative(exits)
            hid_er = out_del @ self.weights_hidden_output.T
            hid_del = hid_er * self.relu_derivative(input_ex)

            self.weights_hidden_output +=input_ex.T @ out_del * self.learning_rate
            self.bios_hidden_output += np.sum(out_del, axis=0) * self.learning_rate
            
            self.weights_input_hidden+= self.normalized(X_normalized_train).T @ hid_del * self.learning_rate

    
    def predict(self, X, y):
        vector_in_hid = self.normalized(X) @ self.weights_input_hidden + self.bios_input_hidden
        input_ex = self.relu(vector_in_hid)
        
        vector_hid_out = input_ex @ self.weights_hidden_output + self.bios_hidden_output
        exits = self.softmax(vector_hid_out)
        
        return round(np.mean(np.argmax(exits, axis=1)==y), 4) * 100
    
    def graphErr(self):
        plt.plot(range(self.epochs), self.errors)
        plt.xlabel('Epochs')
        plt.ylabel('Loss price')
        plt.show()
