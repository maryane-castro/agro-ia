import streamlit as st
import json
import os
import pandas as pd

log_file = "logs/performance_test.log"
metrics_file = "logs/performance_metrics.json"

def load_metrics():
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        return metrics
    else:
        st.warning("Arquivo de métricas não encontrado.")
        return []

def load_logs():
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            logs = f.readlines()
        return logs
    else:
        st.warning("Arquivo de logs não encontrado.")
        return []

st.title("Análise Avançada de Resultados de Inferência YOLO")

option = st.selectbox(
    "Selecione a visualização:",
    ("Métricas de Inferência", "Logs de Processamento")
)

if option == "Métricas de Inferência":
    st.header("Métricas de Inferência")
    
    metrics = load_metrics()
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        st.subheader("Tabela de Métricas")
        st.dataframe(metrics_df)

        # Calcular as médias de tempo e confiança
        avg_inference_time = metrics_df["inference_time"].mean()
        avg_confidence = metrics_df["confidences"].apply(lambda x: sum(x)/len(x) if x else 0).mean()

        # Exibir as médias em formato de texto
        st.subheader("Média das Métricas")
        st.write(f"**Média do Tempo de Inferência**: {avg_inference_time:.4f} segundos")
        st.write(f"**Média da Confiança**: {avg_confidence:.4f}")

        st.download_button(
            label="Baixar Métricas",
            data=json.dumps(metrics, indent=4),
            file_name="performance_metrics.json",
            mime="application/json"
        )

elif option == "Logs de Processamento":
    st.header("Logs de Processamento")

    logs = load_logs()

    if logs:
        st.subheader("Visualização dos Logs")
        log_text = "".join(logs)
        st.text_area("Logs", value=log_text, height=300)

        log_level = st.sidebar.selectbox(
            "Filtrar por nível de log:",
            ["INFO", "ERROR", "WARNING", "DEBUG", "CRITICAL", "ALL"]
        )

        filtered_logs = [log for log in logs if log_level in log or log_level == "ALL"]
        st.text_area(f"Logs filtrados ({log_level})", value="".join(filtered_logs), height=300)

        st.download_button(
            label="Baixar Logs",
            data="".join(logs),
            file_name="performance_test.log",
            mime="text/plain"
        )
