
import numpy as np
import cv2
import matplotlib.pyplot as plt 


ksize = 9
#REPRESENTA O QUÃO ESPICHADO SERÁ A FUNÇÃO GAUSSIANA
sigma = 20

#REPRESENTA A ORIENTAÇÃO QUE O A MASCARÁ DO FILTRO SERÁ PASSADO
theta = 1 * np.pi/5

#Comprimento da onda , define a escala padrão que o filtro vai detectar(linhas mais finas ou mais) 
delta = 1*np.pi/8

#Aspecto da gaussiana, se é mais circular ou elipse
gamma = 1

# Define a simetria
phi = 0


img = cv2.imread("gato.jpg") #lê a imagen 
imagem_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converte para escala de cinza

# Cria filtro de Gabor
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, delta, gamma, phi, ktype = cv2.CV_32F)

# Aplicar filtro na imagem
filtered = cv2.filter2D(imagem_cinza, cv2.CV_8UC3, kernel)

# Mostrar resultados
plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("Imagem Original")
plt.imshow(imagem_cinza, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Kernel Gabor")
plt.imshow(kernel, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Imagem Filtrada")
plt.imshow(filtered, cmap="gray")
plt.axis("off")

plt.show()



