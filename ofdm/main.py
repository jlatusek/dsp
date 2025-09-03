import numpy as np
import matplotlib.pyplot as plt

# Parametry
fs = 500_000  # częstotliwość próbkowania (500 kHz, dużo powyżej Nyquista)
N = 512  # rozmiar IFFT
N_subcarriers = 64  # liczba aktywnych podnośnych
T_symbol = 1e-3  # czas trwania symbolu 1 ms

# Zakres częstotliwości dla danych
f_min, f_max = 30e3, 100e3
freqs = np.fft.fftfreq(N, 1 / fs)  # dostępne częstotliwości IFFT

# Wybieramy indeksy subcarrierów w zadanym zakresie
active_idx = np.where((freqs >= f_min) & (freqs <= f_max))[0][:N_subcarriers]

# Generujemy losowe dane QPSK
bits = np.random.randint(0, 2, (N_subcarriers, 2))
symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)  # mapping QPSK

# Bufor częstotliwości (puste + aktywne nośne)
X = np.zeros(N, dtype=complex)
X[active_idx] = symbols

# Generowanie sygnału czasowego przez IFFT
x_time = np.fft.ifft(X, N) * np.sqrt(N)

# Wybór długości symbolu (można też dodać CP)
t = np.arange(N) / fs

# --- Wizualizacja ---
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t * 1e3, np.real(x_time), label="Re")
plt.plot(t * 1e3, np.imag(x_time), label="Im", alpha=0.7)
plt.title("Symbol OFDM w dziedzinie czasu")
plt.xlabel("Czas [ms]")
plt.ylabel("Amplituda")
plt.legend()

plt.subplot(2, 1, 2)
plt.magnitude_spectrum(x_time, Fs=fs, scale="dB")
plt.title("Widmo sygnału OFDM")
plt.xlabel("Częstotliwość [Hz]")

plt.tight_layout()
plt.show()
