import cv2
import numpy as np
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Configurações principais
# =========================
 # tamanho das imagens
REDIMENSIONAR = (256, 256)    
           
# ângulos do filtro Gabor Theta
GABOR_ORIENTACAO = [60, 90, 120, 150, 180] 

# tamanho da matriz do kernel
GABOR_KERNEL_SIZE = 7
       
# largura do filtro
GABOR_SIGMA = 3.0

# comprimento de onda
GABOR_DELTA = 8

# aspecto (alongamento)
GABOR_GAMMA = 0.5

#=====================================
# número de bins do histograma
HIST_BINS = 192

# semente p/ reprodutibilidade
RANDOM_STATE = 42

# =========================
# Função: Extrair features Gabor
# =========================
def extrair_gabor(img_gray, orientacoes=GABOR_ORIENTACAO):
    feats = []
    img_norm = img_gray.astype(np.float32) / 255.0  # normaliza imagem

    for angulos in orientacoes:
        rad = np.deg2rad(angulos)  # converte graus para radianos
        kernel = cv2.getGaborKernel(
            (GABOR_KERNEL_SIZE, GABOR_KERNEL_SIZE),
            sigma=GABOR_SIGMA,
            theta=rad,
            lambd=GABOR_DELTA,
            gamma=GABOR_GAMMA
        )
        filtered = cv2.filter2D(img_norm, cv2.CV_32F, kernel)

        # Estatísticas básicas do filtro como features
        feats.append(filtered.mean())
        feats.append(filtered.var())

    return np.array(feats, dtype=np.float32)



# =========================
# Função LBP - Local Binary Pattern
# =========================
def extrair_lbp(img_gray, P=8, R=1):
    """
    Extrai LBP (Local Binary Pattern) e retorna o histograma normalizado.
    P = número de vizinhos
    R = raio
    """
    h, w = img_gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # calcula LBP básico (3x3)
    for i in range(R, h-R):
        for j in range(R, w-R):
            centro = img_gray[i, j]
            binario = []
            vizinhos = [
                img_gray[i-1, j-1], img_gray[i-1, j], img_gray[i-1, j+1],
                img_gray[i, j+1], img_gray[i+1, j+1], img_gray[i+1, j],
                img_gray[i+1, j-1], img_gray[i, j-1]
            ]
            for v in vizinhos:
                binario.append(1 if v >= centro else 0)
            
            # converte bits em número decimal (acumulação)
            valor = 0
            for bit in binario:
                valor = valor * 2 + bit
            lbp[i, j] = valor

    # histograma normalizado
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    return hist.astype(np.float32)

# =========================
# Combina Gabor + histograma
# =========================
def extrair_features(img):
    img_resized = cv2.resize(img, REDIMENSIONAR)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    feats_gabor = extrair_gabor(gray)
    feats_lbp   = extrair_lbp(gray)
    
    
    # Junta tudo em um vetor único
    return np.concatenate([feats_gabor, feats_lbp])

# =========================
# Carregar dataset
# =========================
caminho_dataset = "Aerial_Landscapes"
classes = ["Agriculture","Airport","Beach","City","Desert","Forest",
           "Grassland","Highway","Lake","Mountain","Parking","Port",
           "Railway","Residential","River"]

features_list, labels, images_for_display = [], [], []

for label, nome_classe in enumerate(classes):
    pasta = os.path.join(caminho_dataset, nome_classe)
    if not os.path.exists(pasta):
        continue
    for file in os.listdir(pasta):
        if file.lower().endswith((".jpg",".png",".jpeg")):
            img = cv2.imread(os.path.join(pasta, file))
            if img is None:
                continue
            features_list.append(extrair_features(img))
            labels.append(label)
            images_for_display.append(cv2.resize(img, REDIMENSIONAR))

features_list = np.array(features_list, dtype=np.float32)
labels = np.array(labels)
print("Pré-processamento concluído! imagens =", len(features_list))

# =========================
# Divisão treino/teste
# =========================
X_train, X_test, y_train, y_test, imgs_train, imgs_test = train_test_split(
    features_list, labels, images_for_display, test_size=0.3, stratify=labels, random_state=RANDOM_STATE
)

# =========================
# Normalização
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# RandomForest
# =========================
rf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                            max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)

# =========================
# Avaliação
# =========================
y_pred = rf.predict(X_test)
print(f"Acurácia RandomForest: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=classes))

# =========================
# Visualização com rótulos e previsões
# =========================
indices = random.sample(range(len(X_test)), min(10,len(X_test)))
for i in indices:
    img = imgs_test[i]
    cv2.imshow(f"Classe certa: {classes[y_test[i]]} | Previsao do modelo: {classes[y_pred[i]]}",
               cv2.resize(img, (480, 360)))
    cv2.waitKey(0)
cv2.destroyAllWindows()
