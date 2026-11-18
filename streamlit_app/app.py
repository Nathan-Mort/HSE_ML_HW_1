import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("Прогнозирование стоимости автомобиля")

# Загрузка модели 
@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pickle")

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Загрузка данных для EDA 
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')

df = load_data()

# Боковая панель
mode = st.sidebar.radio(
    "Режим",
    ["EDA (графики)", "Предсказание (CSV)", "Предсказание (вручную)", "Коэффициенты модели"]
)

# EDA 
if mode == "EDA (графики)":
    st.header("Разведочный анализ данных")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение цены")
        fig, ax = plt.subplots()
        sns.histplot(df["selling_price"], bins=50, kde=True, ax=ax)
        ax.set_xlabel("selling_price")
        st.pyplot(fig)
        st.caption("Цена сильно скошена вправо - есть дорогие выбросы.")

    with col2:
        st.subheader("Корреляционная матрица")
        num_cols = ["selling_price", "year", "km_driven", "mileage", "engine", "max_power", "seats"]
        df_num = df[num_cols].apply(pd.to_numeric, errors='coerce')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_num[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
        st.caption("max_power (0.69) и year (0.43) - сильнейшие связи с ценой.")

    st.subheader("Цена по категориальным признакам")
    cat_col = st.selectbox("Выберите признак", ["fuel", "transmission", "owner", "seller_type"])
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=cat_col, y="selling_price", showfliers=False, ax=ax)
    ax.set_yscale("log")
    plt.xticks(rotation=30)
    st.pyplot(fig)

# Предсказание по CSV 
elif mode == "Предсказание (CSV)":
    st.header("Загрузите CSV с признаками")
    uploaded = st.file_uploader("CSV-файл", type=["csv"])

    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.write("Загружено строк:", len(new_df))
        st.dataframe(new_df.head())

        try:
            preds = model.predict(new_df)
            result = new_df.copy()
            result["predicted_price"] = preds
            st.success("Предсказания готовы")
            st.dataframe(result)

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать результат CSV", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.info("Убедитесь, что CSV содержит те же колонки, что и train (без selling_price и name).")

# Предсказание вручную 
elif mode == "Предсказание (вручную)":
    st.header("Введите признаки автомобиля")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Год выпуска", min_value=1990, max_value=2024, value=2015)
        km_driven = st.number_input("Пробег (км)", min_value=0, value=50000)
        mileage = st.number_input("Расход (kmpl/kmkg)", min_value=0.0, value=18.0)
    with col2:
        engine = st.number_input("Объём двигателя (CC)", min_value=500, value=1200)
        max_power = st.number_input("Мощность (bhp)", min_value=30.0, value=80.0)
        seats = st.selectbox("Число мест", [2, 4, 5, 6, 7, 8, 9, 10, 14], index=2)

    fuel = st.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
    owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

    if st.button("Предсказать цену"):
        row = pd.DataFrame([{
            "year": year, "km_driven": km_driven, "mileage": mileage,
            "engine": engine, "max_power": max_power, "seats": seats,
            "fuel": fuel, "seller_type": seller_type,
            "transmission": transmission, "owner": owner,
        }])
        pred = model.predict(row)[0]
        st.metric("Предсказанная цена", f"{pred:,.0f}")

# Коэффициенты модели 
elif mode == "Коэффициенты модели":
    st.header("Веса модели")

    # Достаём коэффициенты из пайплайна
    try:
        preprocessor = model.named_steps["preprocessor"]
        coefs = model.named_steps["model"].coef_

        # Имена признаков после препроцессинга
        num_feats = preprocessor.transformers_[0][2]  # числовые
        ohe = preprocessor.named_transformers_["cat"]
        cat_feats = ohe.get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(num_feats) + list(cat_feats)

        coef_series = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(6, len(coef_series) * 0.25)))
        colors = ["green" if v > 0 else "red" for v in coef_series]
        coef_series.plot(kind="barh", color=colors, ax=ax)
        ax.set_xlabel("Коэффициент (влияние на цену)")
        ax.set_title("Веса модели Ridge (стандартизированные признаки)")
        st.pyplot(fig)

        st.caption("Зелёный - увеличивает цену, красный - уменьшает. "
                   "Для OHE-признаков коэффициент - разница относительно базовой категории.")
    except Exception as e:
        st.error(f"Не удалось извлечь коэффициенты: {e}")
        st.info("Убедитесь, что в model.pickle сохранён Pipeline с named_steps['model'] и named_steps['preprocessor'].")
