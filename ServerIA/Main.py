import pandas as pd
import numpy as np
from flask import Flask, abort, jsonify, request
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import pickle
import json


x =  '{ "name":"John", "age":30, "city":"New York"}'
y = json.loads(x)
print(y["age"])
app = Flask(__name__)

def carregaArquivos():
    # Carrega dicionário label
    labelencoder_dictArquivo = open('labelencoder_dict.sav', 'rb')
    labelencoder_dict = pickle.load(labelencoder_dictArquivo)

    # Carrega dicionário onehotencoder
    onehotencoder_dictArquivo = open('onehotencoder_dict.sav', 'rb')
    onehotencoder_dict = pickle.load(onehotencoder_dictArquivo)

    # Carrega modelo
    modelo_arquivo = open('modeloTreinadoMoveMe.sav', 'rb')
    modelo_carregado = pickle.load(modelo_arquivo)
    return labelencoder_dict, onehotencoder_dict, modelo_carregado

#Carregando modelos e dicionários
label_dict, onehot_dict, modelo = carregaArquivos()

def getEncoded(test_data,labelencoder_dict,onehotencoder_dict):
    test_encoded_x = None
    for i in range(0,test_data.shape[1]):
        label_encoder =  labelencoder_dict[i]
        feature = label_encoder.transform(test_data.iloc[:,i])
        feature = feature.reshape(test_data.shape[0], 1)
        onehot_encoder = onehotencoder_dict[i]
        feature = onehot_encoder.transform(feature)
        if test_encoded_x is None:
              test_encoded_x = feature
        else:
              test_encoded_x = np.concatenate((test_encoded_x, feature), axis=1)
    return test_encoded_x

escolha = ['Japanese', 'Excellent', 4, 600, 4]

languages = [{'name': 'isd'}, {'name':'aksdh'}]

def classificaEscolha(escolhaLista, label_dict, onehot_dict, modelo_carregado):
    # Converte valores string em dataframe
    colunasCategoricasPredicao = [{'cozinha': escolhaLista[0], 'votos': escolhaLista[1]}]
    colunasCategoricasPredicaoDic = pd.DataFrame(colunasCategoricasPredicao)

    # Busca dataframe no dicionário
    dicionarioCategoriasPredicaoParte1 = getEncoded(colunasCategoricasPredicaoDic, label_dict, onehot_dict)

    # Transforma partes em numpy array
    PegandoParte2Escolha = np.asarray(escolhaLista[2:])
    Parte2Escolha = PegandoParte2Escolha.reshape(1, 3)
    escolha_para_predicao = np.concatenate((dicionarioCategoriasPredicaoParte1, Parte2Escolha), axis=1)
    return modelo.predict(escolha_para_predicao)

@app.route('/', methods=['POST'])
def realizarPredicao():
    if request.method == 'POST':
        cozinha = request.form['cozinha']
        taxa_votos = request.form['taxa_votos']
        alcance_preco = request.form['alcance_preco']
        classifi_agregada = request.form['classifi_agregada']
        votos = request.form['votos']

        print(cozinha, taxa_votos, alcance_preco, classifi_agregada, votos)
        escolha = [cozinha, taxa_votos, alcance_preco, classifi_agregada, votos]

        label_dict, onehot_dict, modelo = carregaArquivos()
        n = classificaEscolha(escolha, label_dict, onehot_dict, modelo)
        base_busca = pd.read_csv('basereduzida.csv', encoding='latin-1')

        retorno = base_busca.loc[(base_busca['media_preco']) > (230 - 50) & (base_busca['taxa_votos'] == 'Excellent') & (base_busca['cozinha'] == 'Japanese') & (base_busca['votos'] >= (votos - 50)) & (base_busca['alcance_preco'] >= alcance_preco)]

        print(retorno.shape)
    return jsonify({'nome_restaurante': "Lucas"})

#print("Média de preço predição carregada para características buscadas: ", classificaEscolha(escolha, label_dict, onehot_dict, modelo))

if __name__ == '__main__':
    app.run(debug=True, port=8081)