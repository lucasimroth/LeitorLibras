# Libras_reconhecimento.py

# Passo 1: Importar todas as bibliotecas necessárias
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from collections import deque

# Passo 2: Carregar os artefatos do modelo treinado
# Estes são os arquivos que gerou durante o treinamento.
# Certifique-se de que eles estão na mesma pasta que este script.
try:
    modelo = tf.keras.models.load_model('modelo_libras_mao_unica.h5')
    scaler = joblib.load('scaler_mao_unica.pkl')
    label_encoder = joblib.load('label_encoder_mao_unica.pkl')
    print("Modelo, scaler e label encoder carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar os arquivos do modelo: {e}")
    print("Certifique-se de que os arquivos 'modelo_libras_mao_unica.h5', 'scaler_mao_unica.pkl' e 'label_encoder_mao_unica.pkl' estão no mesmo diretório.")
    exit()

# Passo 3: Inicializar os objetos e variáveis principais
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# Variáveis para cálculo de FPS (Frames Por Segundo)
pTime = 0

# Fila para suavização de predições
# Isso ajuda a estabilizar a letra mostrada na tela, evitando que ela pisque
# com predições instáveis. Usamos a média das últimas 10 predições.
predicoes_recentes = deque(maxlen=10)

# Passo 4: Loop principal para captura, processamento e reconhecimento

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:

    print("\nCâmera iniciada. Faça os gestos de LIBRAS para reconhecimento.")
    print("Pressione ESC para sair.")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignorando quadro vazio da câmera.")
            continue

        # Vira a imagem horizontalmente para um efeito de espelho
        img = cv2.flip(img, 1)

        # Converte a imagem de BGR para RGB, pois o MediaPipe usa RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Otimização: Marcar a imagem como não gravável para passar por referência
        img.flags.writeable = False
        results = hands.process(imgRGB)
        img.flags.writeable = True

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Desenha os landmarks e conexões na mão
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                # --- Início da Lógica de Predição ---

                # a. Extrai os pontos da mesma forma que na coleta
                pontos = np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark]).flatten()

                # b. Transforma os dados com o scaler carregado
                # O scaler espera um array 2D, então colocamos nossos pontos dentro de uma lista
                pontos_scaled = scaler.transform([pontos])

                # c. Faça a predição com o modelo Keras
                predicao_prob = modelo.predict(pontos_scaled, verbose=0) # verbose=0 para não poluir o console

                # d. Adiciona a predição de maior probabilidade à fila de suavização
                indice_predito_atual = np.argmax(predicao_prob)
                predicoes_recentes.append(indice_predito_atual)

                # e. Determina a predição final com base na mais frequente na fila
                if predicoes_recentes:
                    indice_predito_final = max(set(predicoes_recentes), key=predicoes_recentes.count)
                    confianca = (predicoes_recentes.count(indice_predito_final) / len(predicoes_recentes)) * 100
                    
                    # f. Traduz o índice de volta para a letra usando o label encoder
                    letra_predita = label_encoder.inverse_transform([indice_predito_final])[0]

                    # g. Mostra o resultado na tela
                    cv2.rectangle(img, (10, 10), (350, 80), (0, 0, 0), -1)
                    texto_predicao = f"Letra: {letra_predita}"
                    texto_confianca = f"Confianca: {confianca:.2f}%"
                    cv2.putText(img, texto_predicao, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    cv2.putText(img, texto_confianca, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Mostra a imagem final com as anotações
        cv2.imshow("Reconhecimento de LIBRAS", img)

        # Verifica se a tecla ESC foi pressionada para sair
        key = cv2.waitKey(5) & 0xFF
        if key == 27:
            print("Encerrando o programa.")
            break

# Passo 5: Liberar recursos ao finalizar
cap.release()
cv2.destroyAllWindows()