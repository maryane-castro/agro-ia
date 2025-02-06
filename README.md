
# AGROIA 🚀

Este repositório contém scripts e configurações para treinamento, inferência e avaliação de modelos de segmentação e detecção, utilizando SAM (Segment Anything Model) e YOLO (You Only Look Once). O projeto abrange desde o download de dados até a avaliação de performance dos modelos.

⚠️ **Status:** Em andamento. Algumas partes do projeto podem estar em desenvolvimento ou sendo atualizadas.

## Estrutura do Projeto 📂

```
├── bestFrame               # Scripts relacionados à extração de frames e vídeos
│   ├── bestFramepaste      # Imagens de frames extraídos
│   ├── bestFrame.py        # Script para análise do melhor frame
│   ├── getJustVideo.py     # Script para extrair apenas o vídeo
│   ├── getVideoAndImgs.py  # Script para extrair vídeo e imagens
│   ├── out_video           # Pasta com vídeos e frames extraídos
│   └── svos                # Scripts e informações auxiliares
├── classModels             # Classes principais para SAM, YOLO e detecção
│   ├── classSAM.py         # Classe para manipulação do modelo SAM
│   ├── classYOLODetection.py  # Classe para detecção usando YOLO
│   ├── classYOLO.py        # Classe para YOLO
├── configs                 # Arquivos de configuração dos modelos
├── images                  # Imagens de exemplo para inferência
├── install-ZED-SDK         # Scripts para instalação e exemplo de uso do ZED SDK
│   ├── readme.md           # Documento com instruções de instalação do ZED SDK
│   └── use-exemple.py      # Exemplo de uso da câmera ZED
├── logs                    # Logs de desempenho dos modelos
├── main.py                 # Arquivo principal para execução do treinamento
├── myGradioUI.py           # Interface de usuário com Gradio
├── myMlflow.py             # Configurações do MLflow
├── README.md               # Documentação do projeto
├── requirements.txt        # Dependências do projeto
├── roboflow                # Scripts para download de datasets e pesos
├── tests                   # Testes para verificar a funcionalidade dos modelos
├── trainModel              # Scripts de treinamento para SAM e YOLO
└── requirements.txt        # Dependências do projeto
```

## Instalação ⚙️

Para rodar este projeto, clone o repositório e instale as dependências utilizando o `requirements.txt`.

1. Clone o repositório:

   ```bash
   git clone <url-do-repositorio>
   cd <nome-do-repositorio>
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Instalação do ZED SDK 🎥

Se você está utilizando a câmera ZED, siga as instruções para instalar o ZED SDK.

1. **Baixar o SDK**: Acesse [ZED SDK](https://www.stereolabs.com/en-br/developers/release) e faça o download da versão adequada para o seu sistema operacional.
   
2. **Instalação no Linux**: Siga os passos de instalação no [Guia de Instalação no Linux](https://www.stereolabs.com/docs/installation/linux).

   - Para instalar o SDK, use o comando:
     ```bash
     chmod +x ZED_SDK_Ubuntu22_cuda11.8_v4.0.0.zstd.run
     ./ZED_SDK_Ubuntu22_cuda11.8_v4.0.0.zstd.run
     ```

3. **Biblioteca Python**: Após a instalação, instale a biblioteca Python para interagir com a câmera:
   ```bash
   pip install pyzed
   ```

## Download dos Modelos 📥

Os modelos treinados podem ser baixados diretamente do Google Drive. Acesse o link abaixo para obter os modelos SAM e YOLO:

[Download dos Modelos](https://drive.google.com/drive/folders/1lXxnISwjFu-YpuL-gBkTvSFYo_5dG4T3?usp=sharing)

Após o download, extraia os arquivos na pasta `checkpoints` do repositório.

## Como Treinar o Modelo 🧠

Para treinar os modelos, você pode executar o script `main.py` ou utilizar os notebooks de treinamento localizados na pasta `trainModel`.

### Treinamento do SAM 🔧
1. Execute o notebook `trainModel/sam/fine_tuning_sam.ipynb`.

### Treinamento do YOLO 📊
1. Execute o notebook `trainModel/yolo-seg-11/fine_tuning_yolo.ipynb`.

## Inferência 🔍

Os notebooks na pasta `inferenceNotebook` permitem que você realize inferências utilizando os modelos treinados.

1. Execute o notebook `inferenceNotebook/inference.ipynb` para realizar a inferência em imagens de entrada.
2. As métricas de desempenho podem ser visualizadas no notebook `inferenceNotebook/metrics.ipynb`.

## Logs e Métricas 📈

As métricas de desempenho dos modelos de detecção e segmentação são salvas nas pastas `logs/performance_metrics_detec.json` e `logs/performance_metrics_seg.json`.

## Testes 🧪

Existem testes para as funções de detecção e segmentação em `tests/testsModelDetection.py` e `tests/testsModelSegmentation.py`, além de um teste para a interface de usuário em `tests/interfaceUI.py`.

