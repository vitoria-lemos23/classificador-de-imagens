import cv2
import numpy as np
import matplotlib.pyplot as plt

def lbp_calcular(img):
    
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    # Percorrer pixels (ignora borda de 1 pixel)
    for i in range(1, h-1):
        for j in range(1, w-1):
            centro = img[i, j]
            binario = []
            # Ordem dos vizinhos (sentido horário)
            vizinhos = [
                img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                img[i, j+1], img[i+1, j+1], img[i+1, j],
                img[i+1, j-1], img[i, j-1]
            ]
            for v in vizinhos:
                binario.append(1 if v >= centro else 0)
            
                # Converte lista de bits para decimal (acumulação)
            valor = 0
            for bit in binario:
                valor = valor * 2 + bit

            lbp[i, j] = valor
            
    return lbp

# Exemplo: carregar imagem
img = cv2.imread("gato.jpg", cv2.IMREAD_GRAYSCALE)
lbp_img = lbp_calcular(img)

# Mostrar resultados
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("LBP")
plt.imshow(lbp_img, cmap="gray")
plt.axis("off")

plt.show()
