import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Настройки
st.set_page_config(layout="wide")
st.title("📊 Информация о моделях")

# Вкладки для разных моделей
tab1, tab2, tab3 = st.tabs(["Детекция лиц с последующей маскировкой детектированной области (Модель YOLO12)", 
                           "Детекция ветрогенераторов (Модель YOLO12)", 
                           "Семантическая сегментация аэрокосмических снимков Земли (Модель Unet)"])

with tab1:
    st.header("Детекция лиц с последующей маскировкой детектированной области (Модель YOLO12)")
    
    # Информация о датасете
    st.subheader("1. Число эпох обучения: 25")
    st.subheader("2. Объем выборки: миллиард")
    st.subheader("3. Метрики")

    metric_options = ['PR-curve', 'All-metrics', 'Confusion matrix', 'Confusion matrix нормализованная']
    selected_metric = st.selectbox("Выберите метрику:", metric_options, key="model1_metric")

    if selected_metric == 'PR-curve':
        st.write('### График PR-curve:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model1/PR_curve.jpg')
    elif selected_metric == 'All-metrics':
        st.write('### All-metrics:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model1/all_metrics.jpg')
    elif selected_metric == 'Confusion matrix нормализованная':
        st.write('### Confusion matrix нормализованная:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model1/Conf_metric_norm.jpg')
    elif selected_metric == 'Confusion matrix':
        st.write('### Confusion matrix:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model1/Conf_metric.jpg')

with tab2:
    st.header("Детекция ветрогенераторов (Модель YOLO12)")
    
    # Информация о датасете
    st.subheader("1. Число эпох обучения: 25")
    st.subheader("2. Объем выборки: миллиард")
    st.subheader("3. Метрики")

    metric_options = ['PR-curve', 'All-metrics', 'Confusion matrix', 'Confusion matrix нормализованная']
    selected_metric = st.selectbox("Выберите метрику:", metric_options, key="model2_metric")

    if selected_metric == 'PR-curve':
        st.write('### График PR-curve:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model2/PR_curve.jpg')
    elif selected_metric == 'All-metrics':
        st.write('### All-metrics:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model2/all_metrics.jpg')
    elif selected_metric == 'Confusion matrix нормализованная':
        st.write('### Confusion matrix нормализованная:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model2/Conf_metric_norm.jpg')
    elif selected_metric == 'Confusion matrix':
        st.write('### Confusion matrix:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model2/Conf_metric.jpg')

with tab3:
    st.header("Семантическая сегментация аэрокосмических снимков Земли (Модель Unet)")
    
    # Информация о датасете
    st.subheader("1. Число эпох обучения: 50")
    st.subheader("2. Объем выборки: миллиард")
    st.subheader("3. Метрики")

    metric_options = ['All-metrics']
    selected_metric = st.selectbox("Выберите метрику:", metric_options, key="model3_metric")

    if selected_metric == 'All-metrics':
        st.write('### All-metrics1:')
        st.image(f'/home/savely/ds_bootcamp/ds-phase-2/Computer-vision-project/images/model3/all_metrics.jpg')
