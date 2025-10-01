import plotly.express as px # type: ignore
import streamlit as st # type: ignore

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

st.plotly_chart(fig)
