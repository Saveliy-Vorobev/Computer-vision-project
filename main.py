import streamlit as st

st.set_page_config(
    page_title="О проекте",
    page_icon="👋",
)

# st.markdown(
#     """
#     <style>
#         button[title^=Exit]+div [data-testid=stImage]{
#             test-align: center;
#             display: block;
#             margin-left: auto;
#             margin-right: auto;
#             width: 100%;
#         }
#     </style>
#     """, unsafe_allow_html=True
# )

# st.markdown(
#     '<h3 style="text-align: center;">Наша команда</h3>',
#     unsafe_allow_html=True
# )


st.image("/home/savely/Desktop/2cbad3a417f6d64c27c96aa26ae13aa8.jpg", caption='Картинка с URL', use_container_width=True)
# left_co, cent_co, last_co = st.columns(3)
# with cent_co:
#     st.image('/home/savely/Downloads/0c8147e6-5841-4281-bc48-9630c65c3fd9.jpeg', width=400)
# st.write('')

st.sidebar.success("Выберите страницу")

st.markdown("""
## Сегментация и детекция изображений на основе моделей YOLA и Unet.

**Авторы:** ИРИНА ЕВСЕЕВА, ВЛАДИМИР БУРОБИН, 
            САУВЕЛИЙ ВОРОБЬЕВ

**Описание:**
- **Страница №1-main**: Навигация 
- **Страница №2**: Информация о моделях
- **Страница №3**: Детекция ветрогенераторов (Модель YOLO12)
- **Страница №4**: Детекция лиц с последующей маскировкой детектированной области (Модель YOLO12)
- **Страница №5**: Семантическая сегментация аэрокосмических снимков Земли (Модель Unet)

Переключайтесь между страницами через левый сайдбар! 
""")

# st.page_link('pages/Сегментация_аэрокосмических_снимков.py', label='🚀 Семантически сегментируем аэрокосмические снимки 🚀')

