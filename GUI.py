import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

class PerceptronGUI:

    wine = load_wine()
    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def to_categorical(y):
        y_cat = np.zeros((len(y), 3))
        for i in range(len(y)):
            y_cat[i, y[i]] = 1
        return y_cat

    y_cat = to_categorical(y_train)

    def __init__(self, master):
        self.master = master
        self.master.title("Perceptron GUI")

        self.label_input = ttk.Label(master, text="Input:")
        self.label_input.grid(row=0, column=0, sticky="w")

        self.entry_input = ttk.Entry(master)
        self.entry_input.grid(row=0, column=1, padx=5, pady=5)

        self.button_train = ttk.Button(master, text="Train", command=self.train)
        self.button_train.grid(row=1, column=0, padx=5, pady=5)

        self.button_predict = ttk.Button(master, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=1, padx=5, pady=5)

        self.label_output = ttk.Label(master, text="Output:")
        self.label_output.grid(row=2, column=0, sticky="w")

        self.label_result = ttk.Label(master, text="")
        self.label_result.grid(row=2, column=1, columnspan=2, padx=5, pady=5)

        self.perceptron = NeuralNetwork(self.X.shape[1], 10, len(set(self.y)), 0.001, 500)
        

    def train(self):
        try:
            self.perceptron.learing(self.X_train, self.y_cat)
            self.perceptron.graphErr()
            messagebox.showinfo("Info", "Perceptron trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def predict(self):
        try:
            #1.364000e+01 3.100000e+00 2.560000e+00 1.520000e+01 1.160000e+02 2.700000e+00 3.030000e+00 1.700000e-01 1.660000e+00 5.100000e+00 9.600000e-01 3.360000e+00 8.450000e+02
            input_text = self.entry_input.get()
            input_data = [float(x.strip()) for x in input_text.split() if x.strip()]
            input_data = np.array(input_data)
            print(input_data)
            if len(input_data) != 13:
                raise ValueError("Input should contain two comma-separated values.")
            result = self.perceptron.predictOne([input_data])
            print(result)
            self.label_result.config(text=str(result[0]))
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = PerceptronGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()