# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# IMPORTAÇÕES DO KERAS
import tensorflow as tf # Importe o TensorFlow como tf
from tensorflow import keras
from tensorflow.keras import layers # Use layers de keras.layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier # Importante para usar Keras com Scikit-learn CV

# 1. Carregar os Dados
nome_arquivo_csv = 'dados_gestos.csv'
try:
    data = pd.read_csv(nome_arquivo_csv, header=None)
except FileNotFoundError:
    print(f"Erro: Arquivo '{nome_arquivo_csv}' não encontrado. Certifique-se de ter coletado os dados.")
    exit()

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

print(f"Total de amostras carregadas: {len(X)}")
print(f"Número de features por amostra: {X.shape[1]}")

# 2. Pré-processamento dos Dados
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

num_classes = len(label_encoder.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, 'scaler_mao_unica.pkl')
print("Scaler salvo como 'scaler_mao_unica.pkl'")
joblib.dump(label_encoder, 'label_encoder_mao_unica.pkl')
print("LabelEncoder salvo como 'label_encoder_mao_unica.pkl'")

# -------------------------------------------------------------
# Definição da Função para Criar o Modelo Keras
# --- Baseado no SEU EXEMPLO, mas adaptado para input_shape ---
# -------------------------------------------------------------
def build_model(): # Renomeei para build_model conforme seu exemplo
    model = keras.Sequential([
        # O input_shape deve ser (63,) para seus 63 features
        # O aviso UserWarning sobre Input() ainda pode aparecer, mas a funcionalidade está ok.
        layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
        layers.Dropout(0.3), # Adicionado dropout conforme sugestão anterior
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax') # num_classes da sua base de dados
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------------------------------------
# SEÇÃO DA CROSS-VALIDATION com KerasClassifier
# --- Substitui a criação manual de folds ---
# -------------------------------------------------------------
print("\n--- Realizando Cross-Validation ---")

# Criar o estimador KerasClassifier
# Passamos a FUNÇÃO build_model para o 'model' do KerasClassifier
# KerasClassifier internamente chamará build_model() para cada fold
# verbose=0 para não exibir o progresso de cada época de cada fold
keras_clf = KerasClassifier(
    model=build_model, # Passa a função aqui
    epochs=100, # Aumentei as épocas para dar mais chance de aprendizado
    batch_size=32,
    verbose=0,
    callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]
)

# Definir a estratégia de validação cruzada (ex: 5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Usando kf = KFold como no seu exemplo

# Avaliar o modelo usando validação cruzada
# Passamos X_scaled e y_categorical (rótulos one-hot encoded)
cv_results = cross_val_score(keras_clf, X_scaled, y_categorical, cv=kf, scoring='accuracy')

# Armazenar as acurácias de cada fold
fold_accuracies = cv_results.tolist() # cross_val_score já retorna um array de acurácias

print(f'\nAcurácias por fold: {fold_accuracies}')
print(f'Acurácia média: {np.mean(fold_accuracies):.4f}')
print(f'Desvio padrão: {np.std(fold_accuracies):.4f}')

# -------------------------------------------------------------
# TREINAMENTO DO MODELO FINAL PARA SALVAR
# (Este modelo será usado no seu reconhecedor em tempo real)
# -------------------------------------------------------------
print("\n--- Treinando o modelo final em todo o dataset para salvar ---")

# Criar e treinar o modelo final em todo o conjunto de dados
final_model = build_model() # Chama a função build_model novamente para a instância final
final_model.fit(X_scaled, y_categorical,
                epochs=100, batch_size=32, verbose=1,
                callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)])

# 5. Salvar o Modelo Keras final
nome_modelo = 'modelo_libras_mao_unica.h5'
final_model.save(nome_modelo)
print(f"\nModelo Keras final salvo como '{nome_modelo}'")