import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
import joblib

def extrair_features_glcm(imagem, distances=[1],angles=[0], levels=256, props = ['contrast','correlation','energy','homogeneity']):
    # Normaliza para valores 0-255 inteiros, como exige GLCM
    img_uint8 = (imagem * 255).astype(np.uint8)
    
    # Calcula a GLCM
    glcm = graycomatrix(img_uint8,
                        distances=[1, 2, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        symmetric=True,
                        normed=True)

    # Extrai propriedades da GLCM
    features = []
    for prop in props:
        feat = graycoprops(glcm, prop)
        features.append(feat.mean()) # média caso haja múltiplos ângulos/distâncias
        
    return np.array(features)

def extrair_features_lbp(imagem, P =8, R=1, method='uniform'):
    
    #calcula o LBP
    imagem_uint8 = (imagem * 255).astype("uint8")
    lbp = local_binary_pattern(imagem_uint8, P, R, method)

    
    #Histograma normalizando do LBP
    (hist,_) = np.histogram(lbp.ravel(), 
                            bins = np.arange(0, P + 3),
                            range = (0, P + 2))
    
    #Normalizar histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    return hist

def extrair_features_histograma(imagem, bins=32):
    
    #Imagem em float [0,1], converter para 0-255
    img_uint8 = (imagem * 255).astype(np.uint8) 
    
    #Histograma normalizado
    hist = cv2.calcHist([img_uint8], [0], None, [bins], [0,256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return hist

# --- Configurações do dataset ---
caminho_dataset = "Aerial_Landscapes"
classes = ["Agriculture","Airport", "Beach", "City", "Desert", "Forest", "Grassland", "Highway","Lake" , "Mountain", "Parking", "Port", "Railway", "Residential", "River"]

# Lista para armazenar imagens, labels e imagens originais
images_originais = [] 
images = []
labels = []

# Percorre cada classe
for label, nome_classe in enumerate(classes):
    caminho_classe = os.path.join(caminho_dataset, nome_classe)
    for filename in os.listdir(caminho_classe):
        if filename.endswith(".jpg"):
            # Ler imagem
            img = cv2.imread(os.path.join(caminho_classe, filename))

            #Guardar a imagem original
            images_originais.append(img)
             # Converter para escala de cinza
            img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        
            #Função para evitar ocupar muita memória
            img_normazizada = img_cinza.astype(np.float32)  / 255.0
            
            # Extrair features GLCM, lbp e histograma
            features_glcm = extrair_features_glcm(img_normazizada)
            features_lbp = extrair_features_lbp(img_normazizada)
            features_hist =  extrair_features_histograma(img_normazizada)
            
            features_juntas = np.hstack([features_glcm,features_lbp, features_hist])
            
            images.append( features_juntas)
            labels.append(label)
            
# Converter listas para arrays Numpy
images = np.array(images, dtype=np.float32) #define float32 explicitamente
labels = np.array(labels)

print("Pré-processamento concluído!")
print("Imagens:", images.shape)
print("Labels:", labels.shape)

indices_images = np.arange(len(images))
X_train, X_test, y_train, y_test, orig_train, orig_test = train_test_split(
    images, labels, images_originais,
    test_size=0.3,
    random_state=42,
    stratify=labels
)

modelo_path = "modelo_radom_forest.pkl"
try:
    # Tenta carregar modelo que foi salvo
    clf = joblib.load(modelo_path)
    print("Modelo carregado com sucesso!")

except:
    # Caso não exista, treina e salva
    
    # Cria o modelo
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Treina
    clf.fit(X_train, y_train)

    # Salvar o modelo
    joblib.dump(clf, "modelo_random_forest.pkl")


    # Salvar as classes
    joblib.dump(classes, "classes.pkl")

    print("Modelo treinado e salvo!")




    if not os.path.exists("classes.pkl"):
        joblib.dump(classes,"classes.pkl")

#--------------------Avaliação-----------------------

y_pred = clf.predict(X_test)

# Acurácia

print("Acurácia: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=classes))


indices_aleatorios = random.sample(range(len(X_test)),10)

for i in indices_aleatorios:
    
    img = orig_test[i]
    label_real = y_test[i]
    label_pred = y_pred[i]
    
    # Redimensiona a imagem para 800x600
    img_redimensionada = cv2.resize(img, (800, 600), interpolation=cv2.INTER_LINEAR)
     
    # Mostrar a imagem com título
    cv2.imshow(f"Classificação Correta: {classes[label_real]} | Predição: {classes[label_pred]}", img_redimensionada)
    cv2.waitKey(0) # espera até apertar qualquer tecla


cv2.destroyAllWindows()
    



