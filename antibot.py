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
# 50% são bots óbvios
qty_pick_bot = np.round(np.random.normal(20, 4, int(QTD_BOTS/2)))
# Os outros 50% jogam com essa opção desligada para não levantar suspeitas
tmp_qtypickbot = np.round(np.random.normal(8, 4,  int(QTD_BOTS/2)))
qty_pick_bot = np.absolute(np.concatenate((tmp_qtypickbot, qty_pick_bot)))
qty_pick_regular = np.absolute(np.round(np.random.normal(8, 4, QTD_REGULAR)))


'''
print("Quantity items:")
print(qty_pick_bot)
print(qty_pick_regular)
'''

#Distribuição normal para média do tempo de reacao entre abrir o corpo do monstro e pegar o item
# 50% são bots óbvios
reaction_pick_bot = np.round(np.random.normal(600, 50, int(QTD_BOTS/2)))
# Os outros 50% jogam com essa opção desligada para não levantar suspeitas
tmp_reactionpickbot = np.round(np.random.normal(1500, 250, int(QTD_BOTS/2)))
reaction_pick_bot = np.absolute(np.concatenate((tmp_reactionpickbot, reaction_pick_bot)))
#Bots podem inserir um limite mínimo para a ação começar, então vamos adotar 1/2 dos dados que usam bot óbvio com um intervalo de +500 milisegundos
reaction_pick_bot =  np.concatenate(((reaction_pick_bot[int(QTD_BOTS/4):], reaction_pick_bot[:int(QTD_BOTS/4)] + 500)))
reaction_pick_regular = np.absolute(np.round(np.random.normal(1500, 250, QTD_REGULAR)))

'''
print("Reaction:")
print(reaction_pick_bot)
print(reaction_pick_regular)
'''

#Distribuição normal para consistência do tempo de reacao entre abrir o corpo do monstro e pegar o item
# 50% são bots óbvios
consistency_pick_bot = np.round(np.random.normal(100, 25, int(QTD_BOTS/2)))
# Os outros 50% jogam com essa opção desligada para não levantar suspeitas
tmp_reactionpickbot = np.round(np.random.normal(2500, 800, int(QTD_BOTS/2)))
consistency_pick_bot = np.absolute(np.concatenate((tmp_reactionpickbot, consistency_pick_bot)))
consistency_pick_regular = np.absolute(np.round(np.random.normal(2500, 800, QTD_REGULAR)))

'''
print("Consistency:")
print(consistency_pick_bot)
print(consistency_pick_regular)
'''

#Distribuição normal para porcentagem da vida ao executar uma cura
# 50% são bots óbvios
avg_percent_life_bot = np.round(np.random.normal(60, 2, int(QTD_BOTS/2)), 1)
# Os outros 50% jogam com essa opção desligada para não levantar suspeitas
# Detalhe Importante:
# Caso o jogador esteja com a opção desligada, todos os outros atributos que são de alguma forma derivados desse aqui devem ser afetados também
# Além disso a ordem deve ser diferente dos atributos relacionados a coleta, pois o jogador pode estar com a opção de coleta desativada mas não a de cura (e vice versa)
# Dessa forma representamos os dados reais com muito mais precisão.
tmp_percentlifebot = np.round(np.random.normal(50, 8, int(QTD_BOTS/2)), 1)
avg_percent_life_bot = np.absolute(np.concatenate((avg_percent_life_bot, tmp_percentlifebot)))
#Bots podem inserir um limite mínimo para a ação começar, então vamos adotar 1/2 dos dados que usam bot óbvio com um limite de 80% da vida
avg_percent_life_bot =  np.concatenate(((avg_percent_life_bot[:int(QTD_BOTS/4)] + 18), avg_percent_life_bot[int(QTD_BOTS/4):]))
avg_percent_life_regular = np.absolute(np.round(np.random.normal(50, 8, QTD_REGULAR), 1))

'''
print("Percent life:")
print(avg_percent_life_bot)
print(avg_percent_life_regular)
'''


#Distribuição normal para média do tempo de reacao entre tomar um dano e se curar com alguma magia, poção ou runa
# 50% são bots óbvios
reaction_heal_bot = np.round(np.random.normal(600, 50, int(QTD_BOTS/2)))
# Os outros 50% jogam com essa opção desligada para não levantar suspeitas
tmp_reactionhealbot = np.round(np.random.normal(2500, 450, int(QTD_BOTS/2)))
reaction_heal_bot = np.absolute(np.concatenate((reaction_heal_bot, tmp_reactionhealbot)))
#Bots podem inserir um limite mínimo para a ação começar, então vamos adotar 1/2 dos dados que usam bot óbvio com um intervalo de +300 milisegundos
reaction_heal_bot =  np.concatenate(((reaction_heal_bot[:int(QTD_BOTS/4)] + 300), reaction_heal_bot[int(QTD_BOTS/4):]))
reaction_heal_regular = np.absolute(np.round(np.random.normal(2500, 450, QTD_REGULAR)))

'''
print("Reaction:")
print(reaction_heal_bot)
print(reaction_heal_regular)
'''

#Distribuição normal para consistência do tempo de reacao entre tomar um dano e se curar com alguma magia, poção ou runa
# 50% são bots óbvios
consistency_heal_bot = np.round(np.random.normal(350, 50, int(QTD_BOTS/2)))
# Os outros 50% jogam com essa opção desligada para não levantar suspeitas
tmp_consistencyhealbot =  400 + np.round(np.random.normal(1500, 500, int(QTD_BOTS/2)))
consistency_heal_bot = np.absolute(np.concatenate((consistency_heal_bot, tmp_consistencyhealbot)))
consistency_heal_regular = 400 + np.absolute(np.round(np.random.normal(1500, 500, QTD_REGULAR)))

'''
print("Consistency:")
print(consistency_heal_bot)
print(consistency_heal_regular)
'''

#Distribuição normal para quantidade de monstros mortos
# 50% são bots óbvios
qty_killed_bot = np.round(np.random.normal(20, 5, int(QTD_BOTS/2)))
# Os outros 50% jogam sem auto-cura e tem mais dificuldade de matar monstros mais rapidamente
tmp_qtykilledbot =  np.round(np.random.normal(12, 4, int(QTD_BOTS/2)))
qty_killed_bot = np.absolute(np.concatenate((qty_killed_bot, tmp_qtykilledbot)))
# Só que alguns jogadores regulares conseguem atingir level maximo e ai passam a matar mesmo os bixos mais fortes com muita facilidade.
# E ainda temos jogadores que vão em monstros bem mais fracos para não ter que gastar tanta poção e ganhar experiência por quantidade.
qty_killed_regular = np.round(np.random.normal(10, 4, int(QTD_REGULAR/2)))
tmp_qntykilledregular = np.round(np.random.normal(20, 5, int(QTD_REGULAR/2)))
qty_killed_regular = np.absolute(np.concatenate((qty_killed_regular, tmp_qntykilledregular)))
# Resumo: Esse atributo possui muito ruído


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