# Libras_reconhecimento_construtor_palavras.py

# Passo 1: Importar todas as bibliotecas necessárias
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from collections import deque
import time # Importamos a biblioteca 'time' para o cooldown

# Passo 2: Carregar os artefatos do modelo treinado
try:
    modelo = tf.keras.models.load_model('modelo_libras_mao_unica.h5')
    scaler = joblib.load('scaler_mao_unica.pkl')
    label_encoder = joblib.load('label_encoder_mao_unica.pkl')
    print("Modelo, scaler e label encoder carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar os arquivos do modelo: {e}")
    exit()

# Passo 3: Inicializar os objetos e variáveis principais
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# --- NOVAS VARIÁVEIS PARA CONSTRUÇÃO DE PALAVRAS ---
palavra_formada = ""
letra_atual_estavel = ""
predicoes_recentes = deque(maxlen=10) # Fila para suavizar a predição atual

# Constantes de controle
LIMIAR_CONFIANCA_ESTAVEL = 85.0 # Confiança mínima para considerar uma letra "estável"
COOLDOWN_ADICAO = 1.5 # Tempo (em segundos) de espera para adicionar outra letra
ultimo_tempo_adicao = 0

# Passo 4: Loop principal para captura, processamento e reconhecimento
with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:

    print("\nCâmera iniciada. Construa palavras com gestos de LIBRAS.")
    print("Use as teclas:")
    print("  - [ESPAÇO] para adicionar a letra atual à palavra.")
    print("  - [C] para corrigir (apagar a última letra).")
    print("  - [D] para deletar a palavra inteira.")
    print("  - [ESC] para sair.")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        h, w, c = img.shape # Altura, largura e canais da imagem
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = hands.process(imgRGB)
        img.flags.writeable = True

        letra_predita = ""
        confianca = 0.0

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                pontos = np.array([[lm.x, lm.y, lm.z] for lm in handLms.landmark]).flatten()
                pontos_scaled = scaler.transform([pontos])
                predicao_prob = modelo.predict(pontos_scaled, verbose=0)
                indice_predito_atual = np.argmax(predicao_prob)
                predicoes_recentes.append(indice_predito_atual)

                if predicoes_recentes:
                    indice_estavel = max(set(predicoes_recentes), key=predicoes_recentes.count)
                    confianca = (predicoes_recentes.count(indice_estavel) / len(predicoes_recentes)) * 100
                    letra_predita = label_encoder.inverse_transform([indice_estavel])[0]
                    
                    # Atualiza a letra estável se a confiança for alta
                    if confianca >= LIMIAR_CONFIANCA_ESTAVEL:
                        letra_atual_estavel = letra_predita
                    else:
                        letra_atual_estavel = "" # Limpa se a confiança cair

        # --- Lógica de Teclas para Manipulação da Palavra ---
        key = cv2.waitKey(5) & 0xFF
        if key == 27: # ESC
            break
        
        # BARRA DE ESPAÇO: Adicionar letra
        if key == 32: # Keycode da barra de espaço é 32
            # Só adiciona se houver uma letra estável e se o cooldown já passou
            if letra_atual_estavel != "" and (time.time() - ultimo_tempo_adicao) > COOLDOWN_ADICAO:
                palavra_formada += letra_atual_estavel
                ultimo_tempo_adicao = time.time()
                print(f"Letra '{letra_atual_estavel}' adicionada. Palavra: {palavra_formada}")

        # Tecla 'C': Corrigir (Backspace)
        if key == ord('c'):
            palavra_formada = palavra_formada[:-1]
            print(f"Última letra removida. Palavra: {palavra_formada}")

        # Tecla 'D': Deletar tudo
        if key == ord('d'):
            palavra_formada = ""
            print("Palavra limpa.")

        # --- Interface Gráfica (GUI) na Tela ---
        # 1. Painel de status da predição atual
        cv2.rectangle(img, (10, 10), (350, 80), (0, 0, 0), -1)
        if letra_predita:
            cor_confianca = (0, 255, 0) if confianca >= LIMIAR_CONFIANCA_ESTAVEL else (0, 165, 255)
            cv2.putText(img, f"Analisando: {letra_predita}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor_confianca, 2)
            cv2.putText(img, f"Confianca: {confianca:.2f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_confianca, 1)

        # 2. Painel da palavra formada
        # Posição na parte inferior da tela
        cv2.rectangle(img, (10, h - 70), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(img, palavra_formada, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

        # 3. Instruções na tela
        cv2.putText(img, "ESP: Add | C: Corrige | D: Deleta", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        cv2.imshow("Construtor de Palavras LIBRAS", img)

# Passo 5: Liberar recursos ao finalizar
cap.release()
cv2.destroyAllWindows()