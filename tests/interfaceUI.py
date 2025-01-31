import streamlit as st
import json
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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
    ("Logs de Processamento", "Métricas de Inferência")
)

if option == "Métricas de Inferência":
    st.header("Métricas de Inferência")

    metrics = load_metrics()
    
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        st.subheader("Tabela de Métricas")
        st.dataframe(metrics_df)

        st.sidebar.subheader("Filtros de Métricas")
        metric_option = st.sidebar.selectbox(
            "Escolha a métrica para visualização:",
            ["Tempo de Inferência", "Número de Detecções"]
        )

        if metric_option == "Tempo de Inferência":
            st.subheader("Distribuição do Tempo de Inferência")
            fig = px.histogram(metrics_df, x="inference_time", nbins=30, title="Distribuição do Tempo de Inferência")
            fig.update_xaxes(title_text="Tempo (segundos)")
            fig.update_yaxes(title_text="Número de Imagens")
            st.plotly_chart(fig)
            
        elif metric_option == "Número de Detecções":
            st.subheader("Distribuição do Número de Detecções")
            fig = px.histogram(metrics_df, x="detections", nbins=30, title="Distribuição do Número de Detecções")
            fig.update_xaxes(title_text="Número de Detecções")
            fig.update_yaxes(title_text="Número de Imagens")
            st.plotly_chart(fig)

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

st.sidebar.title("Configurações Avançadas")
st.sidebar.subheader("Configuração dos Gráficos")
show_grid = st.sidebar.checkbox("Mostrar grade no gráfico", True)

if option == "Métricas de Inferência":
    st.sidebar.subheader("Filtro de Data de Processamento")
    date_filter = st.sidebar.date_input("Escolha a data:", [])
    if date_filter:
        st.sidebar.write(f"Filtrando por data: {date_filter}")
