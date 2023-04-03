import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# import the data
with open("sunspots.txt", "r") as f:
    data = f.readlines()
sunspots = np.array([[0, 0] for i in range(len(data))])
for i in range(len(data)):
    temp = data[i]
    temp = temp[0:len(temp) - 1].split('\t')
    sunspots[i] = [float(j) for j in temp]
sunspots = sunspots.T[1]

plt.subplot(1, 2, 1)
plt.plot(sunspots)
plt.xlabel('time')

# spectrum (fft)
spectrum = abs(fft(sunspots))
plt.subplot(1, 2, 2)
plt.plot(spectrum)
plt.xlabel('freq')
plt.show()

print(np.argmax((spectrum[10:100])) + 10)

# zoom in
plt.plot(spectrum[0:100])
plt.xlabel('freq')
plt.show()
