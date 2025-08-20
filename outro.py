import cv2
import numpy as np
import os
import random
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# Funções de extração de features
# ==========================

def extrair_gabor(img, frequencies=[0.05,0.1,0.2,0.3,0.4], 
                  thetas=[0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    img = img.astype(np.float32)/255.0
    features = []
    for theta in thetas:
        for freq in frequencies:
            kernel = cv2.getGaborKernel((7,7), sigma=4.0, theta=theta, lambd=1/freq, gamma=0.5, psi=0)
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            features.append(filtered.mean())
            features.append(filtered.var())
    return np.array(features, dtype=np.float32)

def extrair_gist(img, n_blocks=4, orientations=[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    img = img.astype(np.float32)/255.0
    h, w = img.shape
    block_h, block_w = h // n_blocks, w // n_blocks
    features = []

    for angle in orientations:
        rad = np.deg2rad(angle)
        kernel = cv2.getGaborKernel((7,7), sigma=4.0, theta=rad, lambd=10, gamma=0.5, psi=0)
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        # Média de cada bloco
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = filtered[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                features.append(block.mean())
    return np.array(features, dtype=np.float32)

def extrair_features_completas(img):
    f_gabor = extrair_gabor(img)
    f_gist = extrair_gist(img)
    return np.concatenate([f_gabor, f_gist])

# ==========================
# Carregamento dataset
# ==========================

caminho_dataset = "Aerial_Landscapes"
classes = ["Agriculture","Airport","Beach","City","Desert","Forest","Grassland",
           "Highway","Lake","Mountain","Parking","Port","Railway","Residential","River"]

features_list = []
labels = []
images_originais = []

for label, nome_classe in enumerate(classes):
    caminho_classe = os.path.join(caminho_dataset, nome_classe)
    if not os.path.exists(caminho_classe):
        print(f"Atenção: pasta {caminho_classe} não existe!")
        continue
    for filename in os.listdir(caminho_classe):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            caminho_img = os.path.join(caminho_classe, filename)
            img = cv2.imread(caminho_img)
            if img is None:
                continue
            images_originais.append(img)
            features_list.append(extrair_features_completas(img))
            labels.append(label)

features_list = np.array(features_list, dtype=np.float32)
labels = np.array(labels)

print(f"Total de imagens carregadas: {len(features_list)}")
print("Shape das features:", features_list.shape)
print("Shape dos labels:", labels.shape)

# ==========================
# Normalização
# ==========================

scaler = StandardScaler()
features_list = scaler.fit_transform(features_list)

# ==========================
# Separação treino/teste
# ==========================

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    features_list, labels, images_originais, test_size=0.3, random_state=42, stratify=labels
)

# ==========================
# GridSearch SVM otimizado
# ==========================

param_grid = {
    'C': [10, 50, 100, 200],
    'gamma': [0.0005, 0.001, 0.002, 0.005, 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("Melhores parâmetros encontrados:", grid.best_params_)

# ==========================
# Treinar SVM final
# ==========================

svm_final = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel='rbf')
svm_final.fit(X_train, y_train)

# ==========================
# Avaliação
# ==========================

y_pred = svm_final.predict(X_test)
print(f"Acurácia SVM: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=classes))


# ==========================
# Visualização
# ==========================

indices_aleatorios = random.sample(range(len(X_test)), min(10, len(X_test)))
for i in indices_aleatorios:
    img = img_test[i]
    label_real = y_test[i]
    label_pred = y_pred[i]

    img_red = cv2.resize(img, (800, 600), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(f"Real: {classes[label_real]} | Pred: {classes[label_pred]}", img_red)
    cv2.waitKey(0)
cv2.destroyAllWindows()
