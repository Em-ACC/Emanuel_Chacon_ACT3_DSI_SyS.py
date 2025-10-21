# ============================================================
# ACTIVIDAD FORMATIVA 3 - FILTROS DIGITALES
# Lenguaje: Python
# Objetivo: Diseñar y evaluar filtros pasa bajos pasa altos y pasa bandas
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, firwin, lfilter, freqz

# ------------------------------------------------------------
# 1 Definicion de la señal de entrada
# ------------------------------------------------------------
fs = 1000  # frecuencia de muestreo Hz
t = np.linspace(0, 1, fs, endpoint=False)  # 1 segundo

# señal compuesta suma de tres senoidales mas ruido
f1, f2, f3 = 5, 50, 200  # frecuencias en Hz
signal = (np.sin(2*np.pi*f1*t) +
          0.5*np.sin(2*np.pi*f2*t) +
          0.2*np.sin(2*np.pi*f3*t))
noise = 0.3 * np.random.randn(len(t))
noisy_signal = signal + noise

# visualizar señal original y con ruido
plt.figure(figsize=(10,4))
plt.plot(t, noisy_signal, label='Señal con ruido')
plt.title('Señal original con ruido')
plt.xlabel('Tiempo s')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------
# 2 Diseño de los filtros digitales
# ------------------------------------------------------------

# parametros de diseno
order = 4
lowcut = 40
highcut = 100

# filtro pasa bajos Butterworth
b_low, a_low = butter(order, 40/(fs/2), btype='low')
# filtro pasa altos Chebyshev tipo I
b_high, a_high = cheby1(order, 0.5, 100/(fs/2), btype='high')
# filtro pasa banda FIR con ventana Hamming
b_band = firwin(numtaps=101, cutoff=[40, 100], fs=fs, pass_zero=False)

# ------------------------------------------------------------
# 3 Aplicacion de los filtros
# ------------------------------------------------------------
filtered_low = lfilter(b_low, a_low, noisy_signal)
filtered_high = lfilter(b_high, a_high, noisy_signal)
filtered_band = lfilter(b_band, 1.0, noisy_signal)

# ------------------------------------------------------------
# 4 Visualizacion de resultados
# ------------------------------------------------------------
plt.figure(figsize=(12,8))
plt.subplot(4,1,1)
plt.plot(t, noisy_signal, color='gray')
plt.title('Señal original con ruido')

plt.subplot(4,1,2)
plt.plot(t, filtered_low, color='b')
plt.title('Filtro pasa bajos Butterworth')

plt.subplot(4,1,3)
plt.plot(t, filtered_high, color='g')
plt.title('Filtro pasa altos Chebyshev I')

plt.subplot(4,1,4)
plt.plot(t, filtered_band, color='r')
plt.title('Filtro pasa banda FIR ventana Hamming')

plt.xlabel('Tiempo s')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5 Analisis en frecuencia
# ------------------------------------------------------------
plt.figure(figsize=(10,6))
for b, a, label in [(b_low, a_low, 'Pasa Bajos'),
                    (b_high, a_high, 'Pasa Altos'),
                    (b_band, 1.0, 'Pasa Banda')]:
    w, h = freqz(b, a, worN=8000)
    plt.plot((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)), label=label)

plt.title('Respuesta en frecuencia de los filtros')
plt.xlabel('Frecuencia Hz')
plt.ylabel('Magnitud dB')
plt.legend()
plt.grid()
plt.show()
