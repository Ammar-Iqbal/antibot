import numpy as np

class AntibotDataset:
  def __init__(self, total_players, bot_rate, seed=None):
    self.total_players = total_players
    self.bot_rate = bot_rate
    np.random.seed(seed)

  def generate(self):
    # Quantidade de jogadores bot na amostra
    bot_players = int(self.total_players * self.bot_rate)

    # Quantidade de jogadores reais na amostra
    real_players = self.total_players - bot_players

    data = self.collected_items(real_players, bot_players)
    data = np.column_stack((data, self.avg_reaction_time_to_collect_item(real_players, bot_players)))
    data = np.column_stack((data, self.std_reaction_time_to_collect_item(real_players, bot_players)))
    data = np.column_stack((data, self.damage_reaction(real_players, bot_players)))
    data = np.column_stack((data, self.mean_reaction_time_damage(real_players, bot_players)))
    data = np.column_stack((data, self.std_reaction_time_damage(real_players, bot_players)))
    data = np.column_stack((data, self.killed_enemies(real_players, bot_players)))
    data = np.column_stack((data, self.hungry(real_players, bot_players)))
    
    target = np.array([0] * real_players + [1] * bot_players)

    self.data = data
    self.target = target

    # Exporta a base para um arquivo antibot.data
    merged = np.column_stack((data, target)) 
    np.savetxt("antibot.data", merged, fmt='%d', delimiter=",")

  @staticmethod
  def collected_items(real_players, bot_players):
    # Distribuição normal para numero de itens coletados
    # 50% são bots óbvios
    qty_pick_bot = np.round(np.random.normal(20, 4, int(bot_players/2)))
    # Os outros 50% jogam com essa opção desligada para não levantar suspeitas
    tmp_qtypickbot = np.round(np.random.normal(8, 4,  int(bot_players/2)))
    qty_pick_bot = np.absolute(np.concatenate((tmp_qtypickbot, qty_pick_bot)))
    qty_pick_regular = np.absolute(np.round(np.random.normal(8, 4, real_players)))

    return np.concatenate((qty_pick_regular, qty_pick_bot))

  @staticmethod
  def avg_reaction_time_to_collect_item(real_players, bot_players):
    #Distribuição normal para média do tempo de reacao entre abrir o corpo do monstro e pegar o item
    # 50% são bots óbvios
    reaction_pick_bot = np.round(np.random.normal(600, 50, int(bot_players/2)))
    # Os outros 50% jogam com essa opção desligada para não levantar suspeitas
    tmp_reactionpickbot = np.round(np.random.normal(1500, 250, int(bot_players/2)))
    reaction_pick_bot = np.absolute(np.concatenate((tmp_reactionpickbot, reaction_pick_bot)))
    #Bots podem inserir um limite mínimo para a ação começar, então vamos adotar 1/2 dos dados que usam bot óbvio com um intervalo de +500 milisegundos
    reaction_pick_bot =  np.concatenate(((reaction_pick_bot[int(bot_players/4):], reaction_pick_bot[:int(bot_players/4)] + 500)))
    reaction_pick_regular = np.absolute(np.round(np.random.normal(1500, 250, real_players)))

    return np.concatenate((reaction_pick_regular, reaction_pick_bot))

  @staticmethod
  def std_reaction_time_to_collect_item(real_players, bot_players):
    #Distribuição normal para consistência do tempo de reacao entre abrir o corpo do monstro e pegar o item
    # 50% são bots óbvios
    consistency_pick_bot = np.round(np.random.normal(100, 25, int(bot_players/2)))
    # Os outros 50% jogam com essa opção desligada para não levantar suspeitas
    tmp_consistencypickbot = np.round(np.random.normal(2500, 800, int(bot_players/2)))
    consistency_pick_bot = np.absolute(np.concatenate((tmp_consistencypickbot, consistency_pick_bot)))
    consistency_pick_regular = np.absolute(np.round(np.random.normal(2500, 800, real_players)))

    return np.concatenate((consistency_pick_regular, consistency_pick_bot))

  @staticmethod
  def damage_reaction(real_players, bot_players):
    #Distribuição normal para porcentagem da vida ao executar uma cura
    # 50% são bots óbvios
    avg_percent_life_bot = np.round(np.random.normal(60, 2, int(bot_players/2)), 1)
    # Os outros 50% jogam com essa opção desligada para não levantar suspeitas
    # Detalhe Importante:
    # Caso o jogador esteja com a opção desligada, todos os outros atributos que são de alguma forma derivados desse aqui devem ser afetados também
    # Além disso a ordem deve ser diferente dos atributos relacionados a coleta, pois o jogador pode estar com a opção de coleta desativada mas não a de cura (e vice versa)
    # Dessa forma representamos os dados reais com muito mais precisão.
    tmp_percentlifebot = np.round(np.random.normal(50, 8, int(bot_players/2)), 1)
    avg_percent_life_bot = np.absolute(np.concatenate((avg_percent_life_bot, tmp_percentlifebot)))
    #Bots podem inserir um limite mínimo para a ação começar, então vamos adotar 1/2 dos dados que usam bot óbvio com um limite de 80% da vida
    avg_percent_life_bot =  np.concatenate(((avg_percent_life_bot[:int(bot_players/4)] + 18), avg_percent_life_bot[int(bot_players/4):]))
    avg_percent_life_regular = np.absolute(np.round(np.random.normal(50, 8, real_players), 1))

    return np.concatenate((avg_percent_life_regular, avg_percent_life_bot))

  @staticmethod
  def mean_reaction_time_damage(real_players, bot_players):
    #Distribuição normal para média do tempo de reacao entre tomar um dano e se curar com alguma magia, poção ou runa
    # 50% são bots óbvios
    reaction_heal_bot = np.round(np.random.normal(600, 50, int(bot_players/2)))
    # Os outros 50% jogam com essa opção desligada para não levantar suspeitas
    tmp_reactionhealbot = np.round(np.random.normal(2500, 450, int(bot_players/2)))
    reaction_heal_bot = np.absolute(np.concatenate((reaction_heal_bot, tmp_reactionhealbot)))
    #Bots podem inserir um limite mínimo para a ação começar, então vamos adotar 1/2 dos dados que usam bot óbvio com um intervalo de +300 milisegundos
    reaction_heal_bot =  np.concatenate(((reaction_heal_bot[:int(bot_players/4)] + 300), reaction_heal_bot[int(bot_players/4):]))
    reaction_heal_regular = np.absolute(np.round(np.random.normal(2500, 450, real_players)))

    return np.concatenate((reaction_heal_regular, reaction_heal_bot))

  @staticmethod
  def std_reaction_time_damage(real_players, bot_players):
    #Distribuição normal para consistência do tempo de reacao entre tomar um dano e se curar com alguma magia, poção ou runa
    # 50% são bots óbvios
    consistency_heal_bot = np.round(np.random.normal(350, 50, int(bot_players/2)))
    # Os outros 50% jogam com essa opção desligada para não levantar suspeitas
    tmp_consistencyhealbot =  400 + np.round(np.random.normal(1500, 500, int(bot_players/2)))
    consistency_heal_bot = np.absolute(np.concatenate((consistency_heal_bot, tmp_consistencyhealbot)))
    consistency_heal_regular = 400 + np.absolute(np.round(np.random.normal(1500, 500, real_players)))

    return np.concatenate((consistency_heal_regular, consistency_heal_bot))

  @staticmethod
  def killed_enemies(real_players, bot_players):
    #Distribuição normal para quantidade de monstros mortos
    # 50% são bots óbvios
    qty_killed_bot = np.round(np.random.normal(20, 5, int(bot_players/2)))
    # Os outros 50% jogam sem auto-cura e tem mais dificuldade de matar monstros mais rapidamente
    tmp_qtykilledbot =  np.round(np.random.normal(12, 4, int(bot_players/2)))
    qty_killed_bot = np.absolute(np.concatenate((qty_killed_bot, tmp_qtykilledbot)))
    # Só que alguns jogadores regulares conseguem atingir level maximo e ai passam a matar mesmo os bixos mais fortes com muita facilidade.
    # E ainda temos jogadores que vão em monstros bem mais fracos para não ter que gastar tanta poção e ganhar experiência por quantidade.
    qty_killed_regular = np.round(np.random.normal(10, 4, int(real_players/2)))
    tmp_qntykilledregular = np.round(np.random.normal(20, 5, int(real_players/2)))
    qty_killed_regular = np.absolute(np.concatenate((qty_killed_regular, tmp_qntykilledregular)))
    # Resumo: Esse atributo possui muito ruído

    return np.concatenate((qty_killed_regular, qty_killed_bot))

  @staticmethod
  def hungry(real_players, bot_players):
    #Distribuição para segundos em que o jogador está satisfeito (comeu food recentemente)
    avg_foodtime_bot = np.absolute(np.round(np.random.normal(800, 300, int(bot_players/2))))
    #Bots podem estar com a opção de comer comida automaticamente desativada
    avg_foodtime_bot =  np.absolute(np.concatenate((np.zeros((int(bot_players/2),), dtype=int), avg_foodtime_bot)))
    #Jogadores regulares costumam esquecer de comer ou costumam comer sempre que acham comida, portanto a distribuição aleatoria é a que melhor se encaixa
    avg_foodtime_regular = np.absolute(np.round(np.random.random(real_players) * 1000))

    return np.concatenate((avg_foodtime_regular, avg_foodtime_bot))
