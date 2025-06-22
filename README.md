# LeitorLibras

Sistema de reconhecimento de gestos em LIBRAS (Língua Brasileira de Sinais) usando visão computacional e aprendizado de máquina.

## Descrição

Este projeto implementa um sistema de reconhecimento de sinais em LIBRAS em tempo real, utilizando:
- OpenCV para captura de vídeo
- MediaPipe para detecção de pontos da mão
- TensorFlow/Keras para classificação dos gestos
- Machine Learning para reconhecimento de letras

## Arquivos do Projeto

- `libras_captura.py` - Script para captura e coleta de dados de gestos
- `model_treinamento.py` - Script para treinamento do modelo de machine learning
- `libras_reconhecimento.py` - Sistema principal de reconhecimento em tempo real
- `dados_gestos.csv` - Dataset com dados dos gestos coletados

## Como Usar

### 1. Captura de Dados
```bash
python libras_captura.py
```

### 2. Treinamento do Modelo
```bash
python model_treinamento.py
```

### 3. Reconhecimento em Tempo Real
```bash
python libras_reconhecimento.py
```

## Controles do Sistema de Reconhecimento

- **Espaço**: Adiciona o caractere reconhecido ao nome
- **Backspace**: Remove o último caractere
- **ç/Ç**: Sair do sistema

## Requisitos

- Python 3.x
- OpenCV
- MediaPipe
- TensorFlow/Keras
- NumPy
- scikit-learn
- joblib

## Instalação das Dependências

```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn joblib
```

## Autor

Desenvolvido como projeto acadêmico para reconhecimento de LIBRAS.
