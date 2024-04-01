import streamlit as st
import pandas as pd
from sklearn import datasets
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
from streamlit_ketcher import st_ketcher

pickle_in = open('MLR_model.pkl', 'rb')
MLR_model = pickle.load(pickle_in)

def welcome():
    return 'welcome all'

def prediction(arr):
    prediction = MLR_model.predict(arr)
    print(prediction)
    return prediction

def main():
    st.set_page_config(page_title="🐻 logBB_prediction")
    st.write("""
    # 🧪 Simple logBB Prediction App
    Это приложение проводит оценку **способности молекул проникать через гематоэнцефалический барьер**!
    """)

    molecule = st.text_input("Ввести молекулу в формате SMILES:")
    smile_code = st_ketcher(molecule)
    st.markdown(f"Smile code: ``{smile_code}``")
    data = {'molec': molecule}
    features = pd.DataFrame(data, index=[0])

    df = pd.read_csv('698_descr.csv')

    ind = df[df['SMILES_uncharge'] == molecule].index
    arr = df.iloc[ind, 3:].to_numpy()

    result = ""

    if st.button("Predict"):
        result = prediction(arr)
    st.success('The output is {}'.format(result))


if __name__ == '__main__':
    main()
    #NC(CC1=CC=CC=C1)C(=O)ON1CCCC1C(=O)O
