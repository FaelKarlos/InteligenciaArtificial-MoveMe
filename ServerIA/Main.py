#Importações para o servidor funcionar
import pandas as pd
import numpy as np
from flask import Flask, abort, jsonify, request, Response # Biblioteca para preparar os web service
import pickle
from sklearn.preprocessing import OneHotEncoder
import json

#Instância do Flask para iniciar métodos do web service
app = Flask(__name__)

#Função que carrega os arquivos de dicionários e modelo salvos
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

#Requisita função para carregar modelos e dicionários
label_dict, onehot_dict, modelo = carregaArquivos()

#Função que mapeia a entrada do usuário em formato correto para predição
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

#Função que recebe a entrada do web service para classificação
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

#Função que publica o web service no verbo POST
#Essa função retornada os dados do restaurante preditos para a solicitação
@app.route('/', methods=['GET'])
def realizarPredicao():
    #Verifica o verbo da requisição
        print("Entrou")
        #Pega os dados da requisição do web service
        cozinha = request.args.get('cozinha')
        taxa_votos = request.args.get('taxavotos')
        alcance_preco = request.args.get('alcancepreco')
        classifi_agregada = request.args.get('classifiagregada')
        votos = request.args.get('votos')

        print("Cozinha: ", cozinha)
        #Converte valores passadas no parâmetro para int, para ser usado no loc()
        convertedClassifi_agregada = float(classifi_agregada)
        convertedVotos = int(votos)
        convertedAlcance_preco = int(alcance_preco)

        #Cria lista com entradas recebidas do web service para classficação
        escolha = [cozinha, taxa_votos, alcance_preco, classifi_agregada, votos]

        #Realizando a classificação
        n = classificaEscolha(escolha, label_dict, onehot_dict, modelo)

        predicao = [n[0]]
        print("Preco medio: ", predicao)

        #Lê a base para retorna uma instância do dataframe
        base_busca = pd.read_csv('basereduzida.csv', encoding='latin-1')

        #Busca um restaurante no base reduzida
        retorno = base_busca.loc[(base_busca['media_preco']) > (predicao[0] - 10) & (base_busca['taxa_votos'] == taxa_votos) & (base_busca['cozinha'] == cozinha) & (base_busca['votos'] >= (convertedVotos - 50)) & (base_busca['alcance_preco'] >= convertedAlcance_preco)]

        #Pega o primeiro elemento encontrado para ser retornado
        df = retorno.loc[0]

       # df.drop(['Restaurant ID'],['Country Code','Address','Locality','Locality Verbose','media_preco', 'Currency', 'Has Table booking','Has Online delivery','Is delivering now','Switch to order menu','Rating color'], inplace=True, axis=1)
        df.drop('Restaurant ID',  inplace=True, axis=0)
        df.drop('Country Code', inplace=True, axis=0)
        df.drop('Address', inplace=True, axis=0)
        df.drop('Locality', inplace=True, axis=0)
        df.drop('Locality Verbose', inplace=True, axis=0)
        df.drop('media_preco', inplace=True, axis=0)
        df.drop('Currency', inplace=True, axis=0)
        df.drop('Has Table booking', inplace=True, axis=0)
        df.drop('Has Online delivery', inplace=True, axis=0)
        df.drop('Is delivering now', inplace=True, axis=0)
        df.drop('Switch to order menu', inplace=True, axis=0)
        df.drop('Rating color', inplace=True, axis=0)
        #

        df.rename(index={
            "Restaurant Name": "nome",
            "City": "cidade",
            "Longitude":"longitude",
            "Latitude":"latitude",
        },
                    inplace=True)
        #Cria um Response que será retornado pelo web service
        resp = Response(response=df.to_json(),
                        status=200,
                        mimetype="application/json")

        #Finaliza a sessão do web service e retorna os dados ao solicitante
        print("Finalizou a requisição!")
        print(resp)
        return (resp)

#Inicia o servidor da aplicação
if __name__ == '__main__':
    app.run(debug=True, port=8081)