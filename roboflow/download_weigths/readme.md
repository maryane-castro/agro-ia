

# Inference Local - Roboflow


## **Como Usar**

1. **Instale as dependências**:  
   Crie um arquivo `requirements.txt` com o conteúdo abaixo e instale com o comando:
   ```bash
   pip install -r requirements.txt
   ```
   ```plaintext
   opencv-python>=4.5.0
   matplotlib>=3.5.0
   supervision>=0.1.0
   python-dotenv>=0.21.0
   ```

2. **Configure as variáveis de ambiente**:  
   Crie um arquivo `.env` no mesmo diretório do script e adicione:
   ```env
   MODEL_ID=<id_do_modelo>
   API_KEY=<sua_chave_api>
   ```


3. **Execute o script**:  
   Certifique-se de que o `.env` está configurado corretamente e rode:
   ```bash
   python script.py
   ```

4. **Resultado**:  
   O script exibirá a imagem anotada em uma janela do Matplotlib.

--- 

## **Observação**

Certifique-se de que o caminho da imagem (`image_path`) é válido e que as credenciais (`MODEL_ID` e `API_KEY`) estão corretas.