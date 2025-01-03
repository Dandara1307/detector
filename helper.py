# helper.py

from ultralytics import YOLO

def load_model(model_path):
    """
    Carrega um modelo YOLO a partir do caminho especificado.

    Parâmetros:
        model_path (str): Caminho para o arquivo do modelo YOLO.

    Retorna:
        Um modelo YOLO carregado.
    """
    model = YOLO(model_path)
    return model

def play_webcam(conf, model):
    """
    Função fictícia para ilustrar o uso do helper.

    Parâmetros:
        conf: confiança do modelo YOLO.
        model: o modelo YOLO carregado.
    """
    # Lógica para utilizar a webcam e fazer a detecção de objetos
    pass

# Outras funções que você possa precisar no helper.py
