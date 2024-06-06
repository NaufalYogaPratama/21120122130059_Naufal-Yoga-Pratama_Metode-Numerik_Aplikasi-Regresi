import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Memuat dataset
url = r'C:\Users\naufa\OneDrive\Documents\Tugas Kuliah\Semester 4\Metode Numerik\Metode Numerik - Aplikasi Regresi\Student_Performance.csv'
data = pd.read_csv(url)

# Memilih kolom yang relevan
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]

# Mendefinisikan variabel independen dan dependen
X = data[['Hours Studied']]
y = data['Performance Index']

# Melakukan log-transformasi pada variabel
X_log = np.log(data['Hours Studied'])
y_log = np.log(data['Performance Index'])

# Membuat model regresi linear pada data yang telah di-log-transformasi
model_pangkat = LinearRegression()
model_pangkat.fit(X_log.values.reshape(-1, 1), y_log)

# Melakukan prediksi
y_pred_log = model_pangkat.predict(X_log.values.reshape(-1, 1))
y_pred = np.exp(y_pred_log)

# Plotting hasil regresi
plt.scatter(data['Hours Studied'], data['Performance Index'], color='blue')
plt.plot(data['Hours Studied'], y_pred, color='red')
plt.title('Regresi Pangkat Sederhana')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.show()

# Menghitung galat RMS
rms_error_pangkat = np.sqrt(mean_squared_error(data['Performance Index'], y_pred))
print(f'Galat RMS (kontol): {rms_error_pangkat}')
