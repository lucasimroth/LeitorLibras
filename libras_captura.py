# coleta_dados.py

# Passo 1: Importar todas as bibliotecas necessárias
import cv2
import mediapipe as mp
import time
import numpy as np
import csv
import os

# Passo 2: Inicializar os objetos e variáveis principais

# Inicializa os utilitários do MediaPipe para desenho e o modelo de mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializa a captura de vídeo pela webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# Variáveis para cálculo de FPS (Frames Por Segundo)
pTime = 0 # Previous Time (Tempo Anterior)

# Define o nome do arquivo CSV para armazenar os dados
# O arquivo será criado no mesmo diretório onde o script for executado
nome_arquivo_csv = 'dados_gestos.csv'

# Passo 3: Loop principal para captura, processamento, visualização e coleta de dados

# Usado 'with' para garantir que os recursos do modelo de mãos sejam liberados corretamente
with mp_hands.Hands(
    # Confiança mínima para a detecção inicial da mão ser considerada um sucesso
    min_detection_confidence=0.7,
    # Confiança mínima para o rastreamento da mão ser considerado um sucesso
    min_tracking_confidence=0.5,
    # Número máximo de mãos a serem detectadas
    max_num_hands=1) as hands:

    print("Câmera iniciada. Posicione a mão na frente da câmera.")
    print("Pressione uma tecla de letra (ex: 'a', 'b', 'c') para salvar o gesto correspondente.")
    print("Pressione 'esc' para sair.")

    while cap.isOpened():
        # Lê um quadro (frame) da câmera
        success, img = cap.read()
        if not success:
            print("Ignorando quadro vazio da câmera.")
            continue

        # Para otimizar, marcamos a imagem como não gravável para passá-la por referência
        img.flags.writeable = False
        # Converte a imagem de BGR (padrão do OpenCV) para RGB (padrão do MediaPipe)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem para detectar as mãos
        results = hands.process(imgRGB)

        # Habilita a escrita na imagem novamente para podermos desenhar nela
        img.flags.writeable = True
        
        # --- INÍCIO DA SEÇÃO DE VISUALIZAÇÃO (AGREGADA DO SEU CÓDIGO) ---

        # Verifica se alguma mão foi detectada
        if results.multi_hand_landmarks:
            # Itera sobre a mão detectada
            for handLms in results.multi_hand_landmarks:
                # Obtém as dimensões da imagem
                h, w, c = img.shape
                
                # Calcula a "distância" (tamanho da mão na tela)
                x0, y0 = int(handLms.landmark[0].x * w), int(handLms.landmark[0].y * h)   # Punho
                x12, y12 = int(handLms.landmark[12].x * w), int(handLms.landmark[12].y * h) # Ponta do dedo médio
                distancia = ((x12 - x0) ** 2 + (y12 - y0) ** 2) ** 0.5
                cv2.putText(img, f"Dist: {int(distancia)}px", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Desenha círculos personalizados em cada ponto de referência (landmark)
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Usa o utilitário do MediaPipe para desenhar as conexões entre os pontos
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        # Calcula e exibe o FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Mostra a imagem final com todas as visualizações
        cv2.imshow("Coleta de Dados - LIBRAS", img)

        # --- FIM DA SEÇÃO DE VISUALIZAÇÃO ---


        # --- INÍCIO DA SEÇÃO DE COLETA DE DADOS (DO MEU CÓDIGO ANTERIOR) ---

        # Aguarda 5ms por uma tecla pressionada
        key = cv2.waitKey(5) & 0xFF

        # Se 'q' for pressionado, encerra o programa
        if key == 27: # 27 é o código ASCII para a tecla 'Esc'
            break
        
        # Se uma tecla de letra for pressionada, salva os dados do gesto
        if key!= 255 and chr(key).isalpha():
            classe_gesto = chr(key).upper()
            print(f"Tecla '{classe_gesto}' pressionada. Salvando pontos de referência...")

            # Garante que uma mão está sendo detectada no momento de salvar
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extrai as coordenadas (x, y, z) de cada um dos 21 pontos e as achata em uma única lista
                    pontos = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    
                    # Adiciona a classe do gesto (a letra pressionada) no início da linha de dados
                    linha_dados = [classe_gesto] + list(pontos)

                    # Abre o arquivo CSV em modo 'append' (adicionar ao final) para não sobrescrever dados antigos
                    with open(nome_arquivo_csv, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(linha_dados)
                    
                    print(f"Dados para o gesto '{classe_gesto}' salvos com sucesso em '{nome_arquivo_csv}'")

# Passo 4: Liberar recursos ao finalizar
cap.release()
cv2.destroyAllWindows()

print(f"\nColeta de dados finalizada. Seus dados estão salvos em '{nome_arquivo_csv}'.")