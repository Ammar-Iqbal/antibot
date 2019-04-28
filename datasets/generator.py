import numpy as np
import distributions as dist


class AntibotDataset:
    def __init__(self, total_players, bot_rate, seed=None):
        self.total_players = total_players
        self.bot_rate = bot_rate
        np.random.seed(seed)

    def generate(self):
        # Quantidade de jogadores bot na amostra
        self.bot_players = round(self.total_players * self.bot_rate)
        # Quantidade de jogadores reais na amostra
        self.real_players = self.total_players - self.bot_players

        columns = (
            self.collected_items(),
            self.avg_time_to_collect_item(),
            self.delta_time_to_collect_item(),
            self.reaction_to_heal(),
            self.avg_reaction_time_to_heal(),
            self.delta_reaction_time_to_heal(),
            self.killed_enemies(),
            self.hungry()
        )

        self.data = np.column_stack(columns)
        self.target = np.concatenate((
            np.array([1] * self.bot_players),
            np.array([0] * self.real_players)
        ))

        return self

    def export_csv(self, fname):
        fcontent = np.column_stack((self.data, self.target))
        np.savetxt(fname, fcontent, fmt='%d', delimiter=",")

        return self

    # Total de itens coletados
    def collected_items(self):
        # Sabemos que metade dos bots coletam os itens que veem pela frente
        some_bots = round(self.bot_players / 2)
        some_bots_items = dist.positive(
            np.random.normal, (20, 4, some_bots)).round()

        # Enquanto a outra metade dos bots é mais seletiva
        other_bots = self.bot_players - some_bots
        other_bots_items = dist.positive(
            np.random.normal, (8, 4, other_bots)).round()

        # Assim como os jogadores reais
        players_items = dist.positive(
            np.random.normal, (8, 4, self.real_players)).round()

        return np.concatenate((some_bots_items, other_bots_items, players_items))

    # Tempo médio entre um item aparecer no cenário e ser coletado
    def avg_time_to_collect_item(self):
        # Sabemos que metade dos bots coletam os itens assim que eles aparecem
        some_bots = round(self.bot_players / 2)
        some_bots_reaction_time = dist.positive(
            np.random.normal, (600, 50, some_bots)).round()

        # E metade desses bots utilizam algum atraso fixo para reagir
        some_bots_half = round(some_bots / 2)
        some_bots_reaction_time[-some_bots_half:] += np.random.uniform(
            1, 10, some_bots_half).round() * 100

        # Enquanto a outra metade dos bots é mais realista
        other_bots = self.bot_players - some_bots
        other_bots_reaction_time = dist.positive(
            np.random.normal, (1500, 250, other_bots)).round()

        # Parecida com os jogadores reais
        players_reaction_time = dist.positive(
            np.random.normal, (1500, 250, self.real_players)).round()

        return np.concatenate((some_bots_reaction_time, other_bots_reaction_time, players_reaction_time))

    # Mediana das diferenças entre os tempos de coleta
    def delta_time_to_collect_item(self):
        # Sabemos que metade dos bots sempre coletam itens assim que eles aparecem
        some_bots = round(self.bot_players / 2)
        some_bots_delta_time = dist.positive(
            np.random.normal, (100, 25, some_bots)).round()

        # Enquanto a outra metade é mais realista
        other_bots = self.bot_players - some_bots
        other_bots_delta_time = dist.positive(
            np.random.normal, (2500, 800, other_bots)).round()

        # Parecida com os jogadores reais
        players_delta_time = dist.positive(
            np.random.normal, (2500, 800, self.real_players)).round()

        return np.concatenate((some_bots_delta_time, other_bots_delta_time, players_delta_time))

    # Mediana da quantidade de vida antes das curas começarem
    def reaction_to_heal(self):
        # Sabemos que metade dos bots possuem um limiar para começar a cura
        some_bots = round(self.bot_players / 2)

        # Onde uma parte tem como estratégia utilizar um limiar alto
        some_bots_half = round(some_bots / 2)
        some_bots_half_heal = dist.positive(
            np.random.normal, (60, 2, some_bots_half)).round()

        # E outra parte tem como estratégia utilizar um limiar baixo
        some_bots_other_half = some_bots - some_bots_half
        some_bots_other_half_heal = dist.positive(
            np.random.normal, (20, 2, some_bots_other_half)).round()

        # Sendo que uma das duas estratégias pode ocorrer em qualquer um desses bots
        some_bots_heal = np.random.permutation(np.concatenate(
            (some_bots_half_heal, some_bots_other_half_heal)))

        # Enquanto a outra metade é mais realista
        other_bots = self.bot_players - some_bots
        other_bots_heal = dist.positive(
            np.random.normal, (50, 8, other_bots)).round()

        # Parecida com os jogadores reais
        players_heal = dist.positive(
            np.random.normal, (50, 8, self.real_players)).round()

        return np.concatenate((some_bots_heal, other_bots_heal, players_heal))

    # Tempo médio entre o último dano sofrido e o início da cura
    def avg_reaction_time_to_heal(self):
        # Sabemos que metade dos bots começam a se curar assim que sofrem dano
        some_bots = round(self.bot_players / 2)
        some_bots_reaction_time = dist.positive(
            np.random.normal, (600, 50, some_bots)).round()

        # E metade desses bots utilizam algum atraso fixo para reagir
        some_bots_half = round(some_bots / 2)
        some_bots_reaction_time[-some_bots_half:] += np.random.uniform(
            1, 10, some_bots_half).round() * 100

        # Enquanto a outra metade é mais realista
        other_bots = self.bot_players - some_bots
        other_bots_reaction_time = dist.positive(
            np.random.normal, (2500, 450, other_bots)).round()

        # Parecida com os jogadores reais
        players_reaction_time = dist.positive(
            np.random.normal, (2500, 450, self.real_players)).round()

        return np.concatenate((some_bots_reaction_time, other_bots_reaction_time, players_reaction_time))

    # Mediana das diferenças entre os tempos de último dano sofrido e início da cura
    def delta_reaction_time_to_heal(self):
        # Sabemos que metade dos bots sempre começam a se curar assim que sofrem dano
        some_bots = round(self.bot_players / 2)
        some_bots_delta_time = dist.positive(
            np.random.normal, (1000, 100, some_bots)).round()

        # Enquanto a outra metade é mais realista
        other_bots = self.bot_players - some_bots
        other_bots_delta_time = dist.positive(
            np.random.normal, (1500, 300, other_bots)).round()

        # Parecida com os jogadores reais
        players_delta_time = dist.positive(
            np.random.normal, (1500, 300, self.real_players)).round()

        return np.concatenate((some_bots_delta_time, other_bots_delta_time, players_delta_time))

    # Total de inimigos mortos
    def killed_enemies(self):
        # Sabemos que metade dos bots matam muitos monstros
        some_bots = round(self.bot_players / 2)
        some_bots_kills = dist.positive(
            np.random.normal, (20, 5, some_bots)).round()

        # Enquanto a outra metade é mais contida
        other_bots = self.bot_players - some_bots
        other_bots_kills = dist.positive(
            np.random.normal, (12, 4, other_bots)).round()

        # E os jogadores reais podem jogar muito mal, muito bem ou estar na média
        players_kills = dist.positive(
            np.random.normal, (12, 8, self.real_players)).round()

        return np.concatenate((some_bots_kills, other_bots_kills, players_kills))

    # Tempo total sentindo fome
    def hungry(self):
        # Sabemos que metade dos bots alimenta o personagem frequentemente
        some_bots = round(self.bot_players / 2)
        some_bots_hungry_time = dist.positive(
            np.random.normal, (60, 10, some_bots)).round()

        # Enquanto a outra metade alimenta o personagem aleatoriamente
        other_bots = self.bot_players - some_bots
        other_bots_hungry_time = np.random.uniform(30, 300, some_bots).round()

        # Assim como os jogadores reais que esquecem da tarefa
        players_hungry_time = np.random.uniform(
            30, 300, self.real_players).round()

        return np.concatenate((some_bots_hungry_time, other_bots_hungry_time, players_hungry_time))
