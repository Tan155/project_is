import streamlit as st


st.set_page_config(page_title="INTELIGENT SYSTEM")
st.title("MATCHINE LEARNING")


if st.button("Machine Learning Knowledge"):
    st.switch_page("pages/Machine_Learning_Knowledge.py")

if st.button("Machine Learning DEMO"):
    st.switch_page("pages/Machine_Learning_Model.py")

st.title("NETURAL NETWORK")

if st.button("Netural Network Knowledge"):
    st.switch_page("pages/Netural_Network Knowledge.py")

if st.button("Netural Network DEMO"):
    st.switch_page("pages/Netural_Network Model.py")
