

# AGROIA 🚀

Este repositório contém scripts e configurações para treinamento, inferência e avaliação de modelos de segmentação e detecção, utilizando SAM (Segment Anything Model) e YOLO (You Only Look Once). O projeto abrange desde o download de dados até a avaliação de performance dos modelos.

⚠️ **Status:** Em andamento. Algumas partes do projeto podem estar em desenvolvimento ou sendo atualizadas.

## Estrutura do Projeto 📂

```
├── checkpoints             # Contém os modelos e pesos salvos durante o treinamento
│   └── v1
│   └── v2
├── classModels             # Classes principais para SAM, YOLO e detecção
│   ├── classSAM.py
│   ├── classYOLO.py
│   └── classYOLODetection.py
├── configs                 # Arquivos de configuração dos modelos
├── images                  # Imagens de exemplo para inferência
├── inferenceNotebook       # Notebooks para inferência e métricas de desempenho
├── logs                    # Logs de desempenho dos modelos
├── main.py                 # Arquivo principal para execução do treinamento
├── mlflow.py               # Configurações do MLflow
├── README.md               # Documentação do projeto
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

## Download dos Modelos 📥

Os modelos treinados podem ser baixados diretamente do Google Drive. Acesse o link abaixo para obter os modelos SAM e YOLO:

[Download dos Modelos](https://drive.google.com/drive/folders/15ZvUu4UkY3lcgn3to8BufOWoz-jL-I2a?usp=sharing)

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

