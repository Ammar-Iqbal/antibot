'Para rodar apenas tenha certeza de ter numpy instalado e digite python antibot.py no console'

#https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html
#https://d1b10bmlvqabco.cloudfront.net/attach/jpi85qyltcw32h/jeybwmqziji5r/jsm9bfxg4bfg/Projeto_AM.pdf
#from random import seed, randint
import numpy as np

# Numa amostra de 100 jogadores, em média 20% são botters.
QTD_TOTAL_JOGADORES = 1000
PROPORTION_BOT = 20
PROPORTION_REGULAR = 80
QTD_BOTS = int(PROPORTION_BOT/100 * QTD_TOTAL_JOGADORES)
QTD_REGULAR = QTD_TOTAL_JOGADORES - QTD_BOTS

''' Ignorar pois saber as sessões para esta atividade não é relevante
# Número máximo e mínimo de sessões por jogador
# (cada sessão considera 5 minutos de jogo)
SESSAO_MIN = 0
SESSAO_MAX = 72 #6 horas seguidas

#Inicializa os arrays de numero de sessões para cada jogador
sessao_bot = np.zeros((QTD_BOTS,), dtype=int)
sessao_regular = np.zeros((QTD_REGULAR,), dtype=int)

TOTAL_ROWS = 0

for v in range(QTD_TOTAL_JOGADORES):
    if v < QTD_BOTS:
        sessao_bot[v] = randint(SESSAO_MAX/6, SESSAO_MAX) #Gerealmente bots jogam pelo menos uma hora
    else:
        sessao_regular[v-QTD_BOTS] = randint(SESSAO_MIN, SESSAO_MAX) #Jogadores regulares não costumam refletir padrões pois varia com perfil de cada um
'''

#Definindo a semente
np.random.seed(255)

#Distribuição normal para numero de itens coletados
qty_pick_bot = np.absolute(np.round(np.random.normal(20, 4, QTD_BOTS)))
qty_pick_regular = np.absolute(np.round(np.random.normal(8, 4, QTD_REGULAR)))


'''
print("Quantity items:")
print(qty_pick_bot)
print(qty_pick_regular)
'''

#Distribuição normal para média do tempo de reacao entre abrir o corpo do monstro e pegar o item
reaction_pick_bot = np.absolute(np.round(np.random.normal(600, 50, QTD_BOTS)))
#Bots podem inserir um limite mínimo para a ação começar, então vamos adotar metade dos dados com um intervalo de 500 milisegundos
reaction_pick_bot =  np.concatenate(((reaction_pick_bot[:int(QTD_BOTS/2)] + 500), reaction_pick_bot[int(QTD_BOTS/2):]))
reaction_pick_regular = np.absolute(np.round(np.random.normal(1500, 250, QTD_REGULAR)))

'''
print("Reaction:")
print(reaction_pick_bot)
print(reaction_pick_regular)
'''

#Distribuição normal para consistência do tempo de reacao entre abrir o corpo do monstro e pegar o item
consistency_pick_bot = np.absolute(np.round(np.random.normal(100, 25, QTD_BOTS)))
consistency_pick_regular = np.absolute(np.round(np.random.normal(2500, 1000, QTD_REGULAR)))

'''
print("Consistency:")
print(consistency_pick_bot)
print(consistency_pick_regular)
'''

#Distribuição normal para porcentagem da vida ao executar uma cura
avg_percent_life_bot = np.absolute(np.round(np.random.normal(60, 2, QTD_BOTS), 1))
#Bots podem inserir um limite mínimo para a ação começar, então vamos adotar metade dos dados com um limite de 80% da vida
avg_percent_life_bot =  np.concatenate(((avg_percent_life_bot[:int(QTD_BOTS/2)] + 18), avg_percent_life_bot[int(QTD_BOTS/2):]))
avg_percent_life_regular = np.absolute(np.round(np.random.normal(50, 8, QTD_REGULAR), 1))

'''
print("Percent life:")
print(avg_percent_life_bot)
print(avg_percent_life_regular)
'''

#Distribuição normal para média do tempo de reacao entre tomar um dano e se curar com alguma magia, poção ou runa
reaction_heal_bot = np.absolute(np.round(np.random.normal(600, 50, QTD_BOTS)))
#Bots podem inserir um limite mínimo para a ação começar, então vamos adotar metade dos dados com um intervalo de +300 milisegundos
reaction_heal_bot =  np.concatenate(((reaction_heal_bot[:int(QTD_BOTS/2)] + 300), reaction_heal_bot[int(QTD_BOTS/2):]))
reaction_heal_regular = np.absolute(np.round(np.random.normal(2500, 450, QTD_REGULAR)))

'''
print("Reaction:")
print(reaction_heal_bot)
print(reaction_heal_regular)
'''

#Distribuição normal para consistência do tempo de reacao entre tomar um dano e se curar com alguma magia, poção ou runa
consistency_heal_bot = np.absolute(np.round(np.random.normal(350, 50, QTD_BOTS)))
consistency_heal_regular = 400 + np.absolute(np.round(np.random.normal(1500, 500, QTD_REGULAR)))

'''
print("Consistency:")
print(consistency_heal_bot)
print(consistency_heal_regular)
'''

#Distribuição normal para quantidade de monstros mortos
qty_killed_bot = np.absolute(np.round(np.random.normal(20, 5, QTD_BOTS)))
qty_killed_regular = np.absolute(np.round(np.random.normal(10, 4, QTD_REGULAR)))

'''
print("Quantity monsters killed:")
print(qty_killed_bot)
print(qty_killed_regular)
'''

#Distribuição para segundos em que o jogador está satisfeito (comeu food recentemente)
avg_foodtime_bot = np.absolute(np.round(np.random.normal(800, 300, int(QTD_BOTS/2))))
#Bots podem estar com a opção de comer comida automaticamente desativada
avg_foodtime_bot =  np.absolute(np.concatenate((np.zeros((int(QTD_BOTS/2),), dtype=int), avg_foodtime_bot)))
#Jogadores regulares costumam esquecer de comer ou costumam comer sempre que acham comida, portanto a distribuição aleatoria é a que melhor se encaixa
avg_foodtime_regular = np.absolute(np.round(np.random.random(QTD_REGULAR) * 1000))

'''
print("Average Food Time:")
print(avg_foodtime_bot)
print(avg_foodtime_regular)
'''

jogador_bot = np.vstack([qty_pick_bot, reaction_pick_bot, consistency_pick_bot, avg_percent_life_bot, reaction_heal_bot, consistency_heal_bot, qty_killed_bot, avg_foodtime_bot]).T
jogador_regular = np.vstack([qty_pick_regular, reaction_pick_regular, consistency_pick_regular, avg_percent_life_regular, reaction_heal_regular, consistency_heal_regular, qty_killed_regular, avg_foodtime_regular]).T

X = np.vstack([jogador_bot, jogador_regular]) # primeiro os bots
Y = np.array([1] * QTD_BOTS + [0] * QTD_REGULAR)


merged = np.column_stack((X,Y)) 

# Exporta nossa base para antibot.data
np.savetxt("antibot.data", merged, fmt='%d', delimiter=",") #%.1f

# Os parametros seguem a seguinte ordem:
#qty_pick_bot, reaction_pick_bot, consistency_pick_bot, avg_percent_life_bot, reaction_heal_bot, consistency_heal_bot, qty_killed_bot, avg_foodtime_bot, preditor da classe (bot = 1, não bot = 0)