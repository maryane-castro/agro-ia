

# Como Instalar o ZED SDK no Linux 💻

Se você deseja usar a câmera ZED no seu projeto Python com a versão 3.11 do Python, siga as etapas abaixo para instalar o ZED SDK no Linux. O processo é simples, mas siga cada etapa com atenção!

## Passos para Instalação 🚀

### 1. Baixar o SDK

Primeiro, faça o **download do ZED SDK** a partir do link abaixo:

🔗 [Baixar o ZED SDK](https://www.stereolabs.com/en-br/developers/release)

Escolha a versão correspondente ao seu sistema operacional.

### 2. Preparar o Sistema 🔧

Antes de continuar, certifique-se de ter **Python 3.6 ou superior** instalado no seu sistema **(Utilizei python 3.11)🚀🚀**.

### 3. Instalar o SDK no Linux 📥

#### 3.1 - Baixar o Instalador

Após o download, vá até o diretório onde o arquivo foi salvo. Abra o terminal e navegue até o local de download:

```bash
cd caminho/para/a/pasta/de/download
```

#### 3.2 - Instalar Dependências 🛠️

Se você não tiver o `zstd` instalado, será necessário instalar antes de continuar. No terminal, execute o comando:

```bash
sudo apt install zstd
```

#### 3.3 - Conceder Permissões de Execução 🎯

Agora, adicione permissões de execução para o arquivo baixado. O nome do arquivo pode variar de acordo com a versão, então substitua "ZED_SDK_Ubuntu22_cuda11.8_v4.0.0.zstd.run" pelo nome exato do arquivo que você baixou.

```bash
chmod +x ZED_SDK_XxxxxXX_cudaxx.x_vx.0.0.zstd.run
```

#### 3.4 - Iniciar a Instalação 🚀

Execute o instalador:

```bash
./ZED_SDK_XxxxxXX_cudaxx.x_vx.0.0.zstd.run
```

**Observação**: No início da instalação, será exibido o contrato de licença. Pressione **q** para aceitar.

Durante o processo de instalação, você será questionado sobre a instalação de dependências, ferramentas e amostras. Responda **y** para "sim" e **n** para "não", conforme necessário. Pressione **Enter** para selecionar as opções padrão.

---

### 4. Configuração Adicional 🧩

#### 4.1 - Instalar a Biblioteca Python 🐍

Depois de instalar o SDK, você precisará instalar a biblioteca Python para usar a câmera ZED. No terminal, execute:

```bash
pip install pyzed
```

Essa biblioteca permite que você acesse as funcionalidades da câmera diretamente no Python.

---

### 5. Documentação e Desenvolvimento 📚

Agora que o SDK está instalado, você pode começar a desenvolver e testar suas aplicações. Para mais detalhes sobre como usar o SDK e explorar as ferramentas disponíveis, consulte a documentação oficial:

- **Documentação Oficial para Linux**: [Instalação no Linux](https://www.stereolabs.com/docs/installation/linux)
- **Desenvolvimento em Python**: [Instalar ZED SDK para Python](https://www.stereolabs.com/docs/app-development/python/install)

### 6. Links Úteis 🔗

- **Página do GitHub**: [ZED SDK no GitHub](https://github.com/stereolabs/zed-sdk?tab=readme-ov-file)
- **Documentação Completa**: [Documentação Completa do SDK](https://www.stereolabs.com/docs)

---


