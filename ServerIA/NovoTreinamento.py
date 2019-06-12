import pandas as pd
import numpy as np
from flask import Flask, abort, jsonify, request, Response # Biblioteca para preparar os web service
import pickle
from sklearn.preprocessing import OneHotEncoder
import json

app = Flask(__name__)

base = pd.read_csv('zomato.csv', encoding='latin-1')

base = base[~base["Cuisines"].str.contains(" ", na=False)]
base['Cuisines'] = base['Cuisines'].fillna("Brazilian")

cozinha = pd.read_csv('cozinhabinario.csv', encoding='utf-8')
cozinhas = pd.read_csv('cozinhas.csv', encoding='utf-8')

numerocozinhas = pd.Series(base['Cuisines'], dtype="category")
colunacozinha = numerocozinhas.cat.rename_categories(np.arange(72))

def carregaArquivos():
    # Carrega dicionário label
    labelencoder_dictArquivo = open('labelencoder_dict_para_cozinha.sav', 'rb')
    labelencoder_dict = pickle.load(labelencoder_dictArquivo)

    # Carrega dicionário onehotencoder
    onehotencoder_dictArquivo = open('onehotencoder_dict_para_cozinha.sav', 'rb')
    onehotencoder_dict = pickle.load(onehotencoder_dictArquivo)

    # Carrega modelo
    modelo_arquivo = open('modelo_para_cozinha.pkl', 'rb')
    modelo_carregado = pickle.load(modelo_arquivo)
    return labelencoder_dict, onehotencoder_dict, modelo_carregado

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

#escolha = ['Excellent', 1000, 4, 4, 600]

def buscarestaurante(predicao):
    for item in range(len(cozinha)):
        if colunacozinha.values[item] == [predicao[0]]:
            #print("Predicao deu: ", [predicao[0]])
            #print("DF:\n ", base.loc[item])
            #retorno = base.loc[item]
            return base.loc[item]

def classificaEscolha(escolhaLista):
    data = {'c1': [escolhaLista[0]], 'c2': [escolhaLista[0]]}

    # Create DataFrame
    datas = pd.DataFrame(data)

    #Valor com encoded
    dicionarioCategoriasPredicaoParte1 = getEncoded(datas, label_dict, onehot_dict)

    PegandoParte2Escolha = np.asarray(escolhaLista[1:])
    Parte2Escolha = PegandoParte2Escolha.reshape(1, 4)
    escolha_para_predicao = np.concatenate((dicionarioCategoriasPredicaoParte1, Parte2Escolha), axis=1)

    predicao = modelo.predict(escolha_para_predicao)

    df = buscarestaurante(predicao)

    return df

@app.route('/m', methods=['GET'])
def predicao():

    taxa_votos = request.args.get('taxavotos')
    mediapreco = request.args.get('mediapreco')
    alcance_preco = request.args.get('alcancepreco')
    classifi_agregada = request.args.get('classificacaoagregada')
    votos = request.args.get('votos')

    # Converte valores passadas no parâmetro para int, para ser usado no loc()
    convertedMediaPreco = int(mediapreco)
    convertedClassifi_agregada = float(classifi_agregada)
    convertedVotos = int(votos)
    convertedAlcance_preco = int(alcance_preco)

    # Cria lista com entradas recebidas do web service para classficação
    escolha = [taxa_votos, convertedMediaPreco, alcance_preco, classifi_agregada, votos]

    df = classificaEscolha(escolha)

    df.drop('Restaurant ID', inplace=True, axis=0)
    df.drop('Country Code', inplace=True, axis=0)
    df.drop('Address', inplace=True, axis=0)
    df.drop('Locality', inplace=True, axis=0)
    df.drop('Locality Verbose', inplace=True, axis=0)
    df.drop('Average Cost for two', inplace=True, axis=0)
    df.drop('Currency', inplace=True, axis=0)
    df.drop('Has Table booking', inplace=True, axis=0)
    df.drop('Has Online delivery', inplace=True, axis=0)
    df.drop('Is delivering now', inplace=True, axis=0)
    df.drop('Rating color', inplace=True, axis=0)
    df.drop('Rating text', inplace=True, axis=0)
    df.drop('Switch to order menu', inplace=True, axis=0)
    df.drop('Votes', inplace=True, axis=0)

    df.rename(index={
        "Restaurant Name": "nome",
        "City": "cidade",
        "Longitude": "longitude",
        "Cuisines": "cozinha",
        "Latitude": "latitude",
        "Price range": "alcance_preco",
        "Aggregate rating": "classificacaoagregada"
    },
        inplace=True)

    resp = Response(response=df.to_json(),
                    status=200,
                    mimetype="application/json")

    return (resp)

if __name__ == '__main__':
    app.run(debug=True, port=8080)