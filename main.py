import os
import pandas as pd
import matplotlib.pyplot as plt
from rl.A2C import A2C
from rl.DQN import DQN
from rl.PPO import PPO
from config.default import RLConfig
from utils.datafilter import load_dataset

# Usado para prevenir o BUG "Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define Configurações Gerais para todas as simulações
config = RLConfig(600, 0.0007, 100000, 0, [128, 128])
print("Definindo Configuração Padrão: {}".format(config))

media_dqn = 0
lista_dqn = []

media_a2c = 0
lista_a2c = []

media_ppo = 0
lista_ppo = []


for _ in range(1):
    # Carrega a base de dados e realiza a separação entre Treinamento (2/3) e Teste (1/3)
    print("Carregando base de Dados.")
    load_dataset(test_size=0.33)
    print("Iniciando Simulações")
    print()

    print("Iniciando Simulação DQN")
    simulacao1 = DQN(1, config=config)
    print("Iniciando Treinamento DQN")
    simulacao1.train()
    print("Iniciando Teste DQN")
    valor_dqn = simulacao1.test()
    lista_dqn.append(valor_dqn)
    media_dqn += valor_dqn
    print()

    print("Iniciando Simulação A2C")
    simulacao2 = A2C(2, config=config)
    print("Iniciando Treinamento A2C")
    simulacao2.train()
    print("Iniciando Teste A2C")
    valor_a2c = simulacao2.test()
    lista_a2c.append(valor_a2c)
    media_a2c += valor_a2c
    print()


    print("Iniciando Simulação PPO")
    simulacao3 = PPO(3, config=config)
    print("Iniciando Treinamento PPO")
    simulacao3.train()
    print("Iniciando Teste PPO")
    valor_ppo = simulacao3.test()
    lista_ppo.append(valor_ppo)
    media_ppo += valor_ppo

    print(100*"=")
    print("DQN")
    print(lista_dqn)
    print(media_dqn/len(lista_dqn))
    print(type(valor_a2c))

    print(100*"=")
    print("A2C")
    print(lista_a2c)
    print(media_a2c/len(lista_a2c))
    print(type(valor_a2c))

    print(100*"=")
    print("PPO")
    print(lista_ppo)
    print(media_ppo/len(lista_ppo))
    print(type(valor_a2c))

print(100*"=")
models=pd.DataFrame(list(zip(lista_dqn,lista_a2c, lista_ppo)), columns=["DQN", "A2C", "PPO"])
print(models)
models.to_csv("teste.csv")

models.index+=1
fig = plt.figure(1)
plt.plot(models['DQN'])
plt.plot(models['A2C'])
plt.plot(models['PPO'])
plt.legend(["DQN", "A2C", "PPO"], loc='upper right')
plt.title("ACURÁCIA DOS MODELOS")
plt.xlabel("NUMERO DE EXECUÇÕES")
plt.ylabel("ACURÁCIA")
fig.salvefig("acurácia.png", dpi=300, bbox_inches='tight')






'''lista1 = ["DQN", lista_dqn]
df = pd.DataFrame(lista1, columns=['lista'])
print(df)'''




