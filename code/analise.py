import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():
    df = pd.read_excel('HIST_PAINEL_COVIDBR_27ago2020.xlsx')
    print(df.dtypes)
    moc  = df[['municipio','codmun','estado', 'semanaEpi', 'casosAcumulado', 'casosNovos', 'obitosNovos']].where(df['codmun'] == 314330)
    moc  = moc.dropna()
    print(moc.corr())
    print(moc[['casosNovos', 'obitosNovos']].corr())
    moc.to_csv('data_moc.csv')

if __name__ == '__main__':
    main()

