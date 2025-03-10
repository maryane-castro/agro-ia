# AGROIA 🚀

Este repositório contém scripts e configurações para treinamento, inferência e avaliação de modelos de segmentação, detecção e regressão. Ele abrange desde o download de dados até a avaliação de performance dos modelos, utilizando abordagens como YOLO e modelos de regressão.

⚠️ **Status:** Em andamento. Algumas partes do projeto podem estar em desenvolvimento ou sendo atualizadas.

## Estrutura do Projeto 📂

```
├── data                   # Dados utilizados no projeto
│   ├── images             # Imagens processadas e brutas
│   ├── processed          # Dados processados
│   └── svos               # Arquivos auxiliares
├── docs                   # Documentação e guias de instalação
│   ├── install-ZED-SDK    # Guia de instalação do SDK da ZED
│   └── roboflow           # Scripts para download de datasets e pesos
├── logs                   # Logs de desempenho e testes
├── models                 # Modelos treinados e suas implementações
│   ├── weights            # Pesos dos modelos treinados
├── scripts                # Scripts auxiliares
│   └── trains             # Scripts de treinamento
├── src                    # Código-fonte principal
│   ├── main.py            # Código principal do projeto
│   └── utils              # Utilitários para processamento e extração de dados
├── tests                  # Testes unitários do projeto
├── README.md              # Documentação do projeto
├── requirements.txt       # Dependências do projeto
└── scripts/setup.sh       # Script de configuração
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
   
2. **Instalação no Linux**:

   ```bash
   chmod +x ZED_SDK_Ubuntu22_cuda11.8_v4.0.0.zstd.run
   ./ZED_SDK_Ubuntu22_cuda11.8_v4.0.0.zstd.run
   ```

3. **Biblioteca Python**:

   ```bash
   pip install pyzed
   ```

4. **Você pode consultar docs/ para mais informações**

## Download dos Modelos 📥

Os modelos treinados podem ser baixados diretamente do Google Drive.

[Download dos Modelos](https://drive.google.com/drive/folders/1lXxnISwjFu-YpuL-gBkTvSFYo_5dG4T3?usp=sharing)

Após o download, extraia os arquivos na pasta `models/weights` do repositório.

## Como Treinar os Modelos 🧠

Para treinar os modelos, utilize os scripts de treinamento na pasta `scripts/trains`.

### Treinamento da Regressão 🔧

1. Utilize `scripts/trains/regression` para treinar os modelos de regressão.
2. Arquivos como `feature_importance.png` e `juntos.csv` auxiliam na análise de features.

### Treinamento do YOLO 📊

1. Execute o notebook `scripts/trains/yolo-seg-11/fine_tuning_yolo.ipynb` para treinar o YOLO.

## Inferência 🔍

Os scripts em `models/class_regression.py` e `models/class_yolo.py` podem ser usados para inferência dos modelos treinados.

1. Execute `src/main.py` para fazer previsões usando os modelos.
2. As previsões podem ser salvas automaticamente em arquivos CSV para análise posterior.

## Logs e Métricas 📈

As métricas de desempenho dos modelos são registradas na pasta `logs/`, incluindo logs de desempenho da detecção e segmentação.



---


