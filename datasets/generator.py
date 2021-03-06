import numpy as np
import distributions as dist


class AntibotDataset:
    def __init__(self, total_players, bot_rate, seed=None):
        self.total_players = total_players
        self.bot_rate = bot_rate
        self.bot_players = round(self.total_players * self.bot_rate)
        self.fair_players = self.total_players - self.bot_players
        np.random.seed(seed)

    def generate(self):
        # Como os bots podem ter características distintas combinadas
        # É feito um embaralhamento das linhas antes de definir as características de um bot

        collection = self.shuffle_and_concatenate((
            self.collected_items(),
            self.avg_time_to_collect_item(),
            self.delta_time_to_collect_item(),
        ))

        healing = self.shuffle_and_concatenate((
            self.heal_threshold(),
            self.avg_time_to_start_healing(),
            self.delta_time_to_start_healing(),
        ))

        kills = self.shuffle_and_concatenate((
            self.killed_enemies(),
        ))

        food = self.shuffle_and_concatenate((
            self.hungry_time(),
        ))

        self.data = np.column_stack((collection, healing, kills, food))
        self.target = np.concatenate((
            np.array([1] * self.bot_players),
            np.array([0] * self.fair_players)
        ))

        return self

    def export_csv(self, fname):
        fcontent = np.column_stack((self.data, self.target))
        np.savetxt(fname, fcontent, fmt='%d', delimiter=",")

        return self

    def shuffle_and_concatenate(self, result):
        pos_sample = tuple([t[0] for t in result])
        neg_sample = tuple([t[1] for t in result])

        pos_group = np.random.permutation(np.column_stack(pos_sample))
        neg_group = np.column_stack(neg_sample)

        return np.concatenate((pos_group, neg_group))

    # Total de itens coletados
    def collected_items(self):
        # Sabemos que metade dos bots coletam itens automaticamente
        some_bots = round(self.bot_players / 2)
        some_bots_collected_items = dist.positive(
            np.random.normal, (20, 4, some_bots)).round()

        # Enquanto a outra metade dos bots não utiliza este recurso
        other_bots = self.bot_players - some_bots
        other_bots_collected_items = dist.positive(
            np.random.normal, (8, 4, other_bots)).round()

        # Assim como os jogadores justos
        players_collected_items = dist.positive(
            np.random.normal, (8, 4, self.fair_players)).round()

        return (np.concatenate((some_bots_collected_items, other_bots_collected_items)), players_collected_items)

    # Tempo médio entre um item aparecer no cenário e ser coletado
    def avg_time_to_collect_item(self):
        # Sabemos que metade dos bots coletam itens automaticamente assim que estes aparecem no cenário
        some_bots = round(self.bot_players / 2)
        some_bots_avg_time = dist.positive(
            np.random.normal, (600, 50, some_bots)).round()

        # E metade desses bots utilizam algum atraso fixo para reagir
        half_some_bots = round(some_bots / 2)
        some_bots_avg_time[half_some_bots:] += np.random.uniform(
            1, 10, half_some_bots).round() * 100

        # Enquanto a outra metade dos bots não utiliza o recurso
        other_bots = self.bot_players - some_bots
        other_bots_avg_time = dist.positive(
            np.random.normal, (1500, 250, other_bots)).round()

        # Assim como os jogadores justos
        players_avg_time = dist.positive(
            np.random.normal, (1500, 250, self.fair_players)).round()

        return (np.concatenate((some_bots_avg_time, other_bots_avg_time)), players_avg_time)

    # Mediana das diferenças entre os tempos de coleta
    def delta_time_to_collect_item(self):
        # Sabemos que metade dos bots coletam itens automaticamente sempre que estes aparecem no cenário
        some_bots = round(self.bot_players / 2)
        some_bots_delta_time = dist.positive(
            np.random.normal, (100, 25, some_bots)).round()

        # Enquanto a outra metade dos trapaceiros não utiliza o recurso
        other_bots = self.bot_players - some_bots
        other_bots_delta_time = dist.positive(
            np.random.normal, (2500, 800, other_bots)).round()

        # Assim como os jogadores justos
        players_delta_time = dist.positive(
            np.random.normal, (2500, 800, self.fair_players)).round()

        return (np.concatenate((some_bots_delta_time, other_bots_delta_time)), players_delta_time)

    # Mediana da quantidade de vida antes das curas começarem
    def heal_threshold(self):
        # Sabemos que metade dos bots possui um limiar alto para começar a cura
        some_bots = round(self.bot_players / 2)
        some_bots_heal_threshold = dist.positive(
            np.random.normal, (60, 2, some_bots)).round()

        # Enquanto a outra metade utiliza um limiar baixo para começar a cura
        other_bots = self.bot_players - some_bots
        other_bots_heal_threshold = dist.positive(
            np.random.normal, (20, 2, other_bots)).round()

        # Ao passo que os jogadores justos são imprevisíveis
        players_heal_threshold = np.random.uniform(10, 95,  self.fair_players).round()

        return (np.concatenate((some_bots_heal_threshold, other_bots_heal_threshold)), players_heal_threshold)

    # Tempo médio entre o último dano sofrido e o início da cura
    def avg_time_to_start_healing(self):
        # Sabemos que metade dos bots começam a se curar assim que sua vida chega a um limiar
        some_bots = round(self.bot_players / 2)
        some_bots_avg_time = dist.positive(
            np.random.normal, (600, 50, some_bots)).round()

        # E metade desses bots utilizam algum atraso fixo para reagir
        other_bots = self.bot_players - some_bots
        other_bots_avg_time = dist.positive(
            np.random.normal, (600, 50, other_bots)).round()
        other_bots_avg_time += np.random.uniform(
            1, 10, other_bots).round() * 100

        bots_avg_time = np.random.permutation(np.concatenate(
            (some_bots_avg_time, other_bots_avg_time)))

        # Ao passo que os jogadores justos variam em torno de 2500 ms
        players_avg_time = dist.positive(
            np.random.normal, (2500, 450, self.fair_players)).round()

        return (bots_avg_time, players_avg_time)

    # Mediana das diferenças entre os tempos de último dano sofrido e início da cura
    def delta_time_to_start_healing(self):
        # Sabemos que metade dos bots sempre começam a se curar assim que sua vida chega a um limiar
        some_bots = round(self.bot_players / 2)
        some_bots_delta_time = dist.positive(
            np.random.normal, (1000, 100, some_bots)).round()

        # Enquanto a outra metade tem um comportamento errático
        other_bots = self.bot_players - some_bots
        other_bots_delta_time = np.random.uniform(
            500, 2000, other_bots).round()

        bots_delta_time = np.random.permutation(np.concatenate(
            (some_bots_delta_time, other_bots_delta_time)))

        # Ao passo que os jogadores justos são previsíveis apesar de variar bastante
        players_delta_time = dist.positive(
            np.random.normal, (1500, 300, self.fair_players)).round()

        return (bots_delta_time, players_delta_time)

    # Total de inimigos mortos
    def killed_enemies(self):
        # Sabemos que metade dos bots matam muitos monstros
        some_bots = round(self.bot_players / 2)
        some_bots_killed_enemies = dist.positive(
            np.random.normal, (20, 5, some_bots)).round()

        # Enquanto a outra metade não usa todo seu potencial
        other_bots = self.bot_players - some_bots
        other_bots_killed_enemies = dist.positive(
            np.random.normal, (12, 4, other_bots)).round()

        # E existem jogadores justos desde os muito ruins até os muito bons
        players_killed_enemies = dist.positive(
            np.random.normal, (12, 8, self.fair_players)).round()

        return (np.concatenate((some_bots_killed_enemies, other_bots_killed_enemies)), players_killed_enemies)

    # Tempo total sentindo fome
    def hungry_time(self):
        # Sabemos que metade dos bots alimenta o personagem frequentemente
        some_bots = round(self.bot_players / 2)
        some_bots_hungry_time = dist.positive(
            np.random.normal, (60, 10, some_bots)).round()

        # Enquanto a outra metade alimenta o personagem aleatoriamente
        other_bots = self.bot_players - some_bots
        other_bots_hungry_time = np.random.uniform(30, 300, other_bots).round()

        # Assim como os jogadores justos, que esquecem de realizar a tarefa
        players_hungry_time = np.random.uniform(
            30, 300, self.fair_players).round()

        return (np.concatenate((some_bots_hungry_time, other_bots_hungry_time)), players_hungry_time)
