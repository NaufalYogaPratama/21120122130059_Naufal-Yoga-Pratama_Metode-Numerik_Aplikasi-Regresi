import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Memuat dataset
url = 'C:\\Users\\naufa\\OneDrive\\Documents\\Tugas Kuliah\\Semester 4\\Metode Numerik\\Metode Numerik - Aplikasi Regresi\\Student_Performance.csv'
data = pd.read_csv(url)

# Memilih kolom yang relevan
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]
X = data[['Hours Studied']]
y = data['Performance Index']

# Model regresi linear
linear_model = LinearRegression()
linear_model.fit(X, y)

# Prediksi
y_pred_linear = linear_model.predict(X)

# Plotting
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred_linear, color='red')
plt.title('Regresi Linear')
plt.xlabel('Jam Belajar')
plt.ylabel('Indeks Performa')
plt.show()

# Menghitung galat RMS
rms_error_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
print(f'Galat RMS (Linear): {rms_error_linear}')
