"""
Joguinho de Reinforcement Learning: Grid World 5x5 com Q-Learning
O agente aprende a navegar da posição inicial até o prêmio.
Janela grafica: ambiente, agente em movimento e recompensa.
"""

import numpy as np
import random
from typing import Tuple, List
import tkinter as tk
from tkinter import font as tkfont

# ============== CONFIGURAÇÃO DO AMBIENTE ==============

GRID_SIZE = 5
START = (0, 0)      # Canto superior esquerdo
GOAL = (4, 4)       # Canto inferior direito (prêmio)

# Acoes: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = ['Up', 'Down', 'Left', 'Right']
N_ACTIONS = 4

# Recompensas
REWARD_GOAL = 100
REWARD_STEP = -1
REWARD_WALL = -5


def state_to_idx(row: int, col: int) -> int:
    """Converte coordenada (row, col) em índice de estado (0 a 24)."""
    return row * GRID_SIZE + col


def idx_to_state(idx: int) -> Tuple[int, int]:
    """Converte índice de estado em (row, col)."""
    return idx // GRID_SIZE, idx % GRID_SIZE


class GridWorld:
    """Ambiente grid 5x5. O agente se move até alcançar o prêmio."""

    def __init__(self):
        self.n_states = GRID_SIZE * GRID_SIZE
        self.state = START
        self.done = False

    def reset(self) -> int:
        """Reseta o ambiente e retorna o estado inicial."""
        self.state = START
        self.done = False
        return state_to_idx(*self.state)

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Executa uma ação. Retorna (próximo_estado, recompensa, done).
        action: 0=Up, 1=Down, 2=Left, 3=Right
        """
        row, col = self.state

        if action == 0:   # Up
            new_row, new_col = row - 1, col
        elif action == 1: # Down
            new_row, new_col = row + 1, col
        elif action == 2: # Left
            new_row, new_col = row, col - 1
        else:              # Right
            new_row, new_col = row, col + 1

        # Bateu na parede?
        if new_row < 0 or new_row >= GRID_SIZE or new_col < 0 or new_col >= GRID_SIZE:
            reward = REWARD_WALL
            # Estado não muda
        else:
            self.state = (new_row, new_col)
            if self.state == GOAL:
                reward = REWARD_GOAL
                self.done = True
            else:
                reward = REWARD_STEP

        next_state = state_to_idx(*self.state)
        return next_state, reward, self.done

    def render(self, agent_pos: Tuple[int, int] = None):
        """Imprime o grid no terminal. agent_pos sobrescreve estado atual se passado."""
        pos = agent_pos if agent_pos is not None else self.state
        print("\n" + "+" + "-" * (GRID_SIZE * 4 - 1) + "+")
        for r in range(GRID_SIZE):
            line = "|"
            for c in range(GRID_SIZE):
                if (r, c) == GOAL:
                    cell = " * "   # premio
                elif (r, c) == pos:
                    cell = " A "   # agente
                else:
                    cell = " . "
                line += cell + "|"
            print(line)
            print("+" + "-" * (GRID_SIZE * 4 - 1) + "+")
        print()


# ============== AGENTE Q-LEARNING ==============

class QLearningAgent:
    """Agente que aprende via Q-Learning (tabela Q)."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Tabela Q: Q[estado, ação]
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state: int, training: bool = True) -> int:
        """Escolhe ação: epsilon-greedy (exploração vs exploração)."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Atualiza Q(s,a) com a regra do Q-Learning."""
        best_next = np.max(self.Q[next_state]) if not done else 0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error

    def decay_epsilon(self):
        """Reduz epsilon após cada episódio (menos exploração com o tempo)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============== TREINAMENTO E AVALIAÇÃO ==============

def train(
    env: GridWorld,
    agent: QLearningAgent,
    n_episodes: int = 500,
    max_steps: int = 100,
    verbose_interval: int = 50,
) -> List[int]:
    """Treina o agente e retorna lista de passos por episódio."""
    steps_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = agent.choose_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        agent.decay_epsilon()
        steps_per_episode.append(steps)

        if (episode + 1) % verbose_interval == 0:
            print(f"Episodio {episode + 1:4d} | Passos: {steps:3d} | "
                  f"Recompensa: {total_reward:5.0f} | eps = {agent.epsilon:.3f}")

    return steps_per_episode


def play_episode(env: GridWorld, agent: QLearningAgent, max_steps: int = 50, render: bool = True) -> int:
    """Roda um episódio sem explorar (só política aprendida). Retorna número de passos."""
    state = env.reset()
    steps = 0
    if render:
        print("Início do episódio:")
        env.render(idx_to_state(state))

    for _ in range(max_steps):
        action = agent.choose_action(state, training=False)
        next_state, reward, done = env.step(action)
        steps += 1
        if render:
            print(f"Passo {steps}: ação {ACTIONS[action]}")
            env.render(idx_to_state(next_state))
        state = next_state
        if done:
            if render:
                print(">>> Agente alcancou o premio!\n")
            break
    return steps


def print_policy(agent: QLearningAgent):
    """Imprime a política aprendida (melhor ação por estado) em formato de grid."""
    print("\nPolitica aprendida (melhor acao por celula):")
    print("+" + "-" * (GRID_SIZE * 5) + "+")
    for r in range(GRID_SIZE):
        line = "|"
        for c in range(GRID_SIZE):
            idx = state_to_idx(r, c)
            if idx_to_state(idx) == GOAL:
                line += "  *  |"
            else:
                a = int(np.argmax(agent.Q[idx]))
                line += f" {ACTIONS[a]:4s} |"
        print(line)
        print("+" + "-" * (GRID_SIZE * 5) + "+")
    print()


# ============== JANELA GRAFICA (AMBIENTE + AGENTE + RECOMPENSA) ==============

CELL_SIZE = 72
GRID_PAD = 2
AGENT_COLOR = "#4A90D9"      # azul (agente)
GOAL_COLOR = "#E8B923"       # dourado (premio/baú)
GRID_BG = "#F0F4F8"
GRID_LINE = "#2C3E50"
TEXT_BG = "#1A252F"
TEXT_FG = "#ECF0F1"


class GridWorldGUI:
    """Janela que mostra o ambiente, o agente se movendo e a recompensa (como no diagrama)."""

    def __init__(self, env: GridWorld, agent: QLearningAgent):
        self.env = env
        self.agent = agent
        self.root = tk.Tk()
        self.root.title("Q-Learning - Grid World 5x5")
        self.root.resizable(False, False)
        self.root.configure(bg=TEXT_BG)

        # Canvas do grid (ambiente)
        size = GRID_SIZE * CELL_SIZE + GRID_SIZE * GRID_PAD + 20
        self.canvas = tk.Canvas(
            self.root, width=size, height=size,
            bg=GRID_BG, highlightthickness=0
        )
        self.canvas.pack(pady=(15, 10), padx=15)

        # Labels: Estado, Acao, Recompensa (referencia ao diagrama)
        info_font = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.lbl_state = tk.Label(
            self.root, text="Estado: (0, 0)",
            font=info_font, fg=TEXT_FG, bg=TEXT_BG
        )
        self.lbl_state.pack(pady=2)
        self.lbl_action = tk.Label(
            self.root, text="Acao: --",
            font=info_font, fg=TEXT_FG, bg=TEXT_BG
        )
        self.lbl_action.pack(pady=2)
        self.lbl_reward = tk.Label(
            self.root, text="Recompensa: 0",
            font=info_font, fg="#F1C40F", bg=TEXT_BG
        )
        self.lbl_reward.pack(pady=2)
        self.lbl_total = tk.Label(
            self.root, text="Recompensa total: 0  |  Passo: 0",
            font=info_font, fg="#2ECC71", bg=TEXT_BG
        )
        self.lbl_total.pack(pady=(2, 8))
        formula_font = tkfont.Font(family="Consolas", size=9)
        self.lbl_formula = tk.Label(
            self.root,
            text="Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]",
            font=formula_font, fg="#BDC3C7", bg=TEXT_BG
        )
        self.lbl_formula.pack(pady=(0, 15))

        self._draw_grid_and_goal()

    def _cell_rect(self, row: int, col: int):
        """Retorna (x1, y1, x2, y2) do retangulo da celula (row, col)."""
        x = 10 + col * (CELL_SIZE + GRID_PAD)
        y = 10 + row * (CELL_SIZE + GRID_PAD)
        return x, y, x + CELL_SIZE, y + CELL_SIZE

    def _draw_grid_and_goal(self):
        """Desenha o grid e o premio (estrela/baú)."""
        self.canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1, y1, x2, y2 = self._cell_rect(r, c)
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill="white", outline=GRID_LINE, width=2
                )
                if (r, c) == GOAL:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_oval(
                        cx - 18, cy - 18, cx + 18, cy + 18,
                        fill=GOAL_COLOR, outline=GRID_LINE, width=2
                    )
                    self.canvas.create_text(cx, cy, text="*", font=("Arial", 22, "bold"), fill="white")

    def _draw_agent(self, row: int, col: int):
        """Desenha o agente (circulo azul) na celula (row, col)."""
        self.canvas.delete("agent")
        x1, y1, x2, y2 = self._cell_rect(row, col)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        self.canvas.create_oval(
            cx - 20, cy - 20, cx + 20, cy + 20,
            fill=AGENT_COLOR, outline=GRID_LINE, width=2, tags="agent"
        )
        self.canvas.create_text(cx, cy, text="A", font=("Arial", 14, "bold"), fill="white", tags="agent")

    def _update_info(self, state: Tuple[int, int], action: str, reward: float, total: float, step: int):
        """Atualiza os textos de estado, acao e recompensa."""
        self.lbl_state.config(text=f"Estado: {state}")
        self.lbl_action.config(text=f"Acao: {action}")
        self.lbl_reward.config(text=f"Recompensa: {reward:+.0f}")
        self.lbl_total.config(text=f"Recompensa total: {total:+.0f}  |  Passo: {step}")

    def run_episode_animated(self, delay_ms: int = 600, max_steps: int = 50):
        """Roda um episodio e anima o agente na tela, atualizando recompensa a cada passo."""
        state = self.env.reset()
        total_reward = 0
        step = 0
        self._draw_grid_and_goal()
        self._draw_agent(*idx_to_state(state))
        self._update_info(idx_to_state(state), "--", 0, 0, 0)

        def next_step():
            nonlocal state, total_reward, step
            if step >= max_steps:
                self.lbl_total.config(text=f"Fim (max passos) | Total: {total_reward:+.0f}")
                return
            action = self.agent.choose_action(state, training=False)
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            step += 1
            self._draw_agent(*idx_to_state(next_state))
            self._update_info(idx_to_state(next_state), ACTIONS[action], reward, total_reward, step)
            state = next_state
            if done:
                self.lbl_total.config(
                    text=f"Premio alcancado! | Total: {total_reward:+.0f} | Passos: {step}"
                )
                return
            self.root.after(delay_ms, next_step)

        self.root.after(delay_ms, next_step)

    def start(self):
        """Inicia o loop da janela."""
        self.root.mainloop()


def open_visualization(env: GridWorld, agent: QLearningAgent, episode_delay_ms: int = 600):
    """Abre a janela grafica e anima um episodio (agente, ambiente, recompensa)."""
    gui = GridWorldGUI(env, agent)
    gui.run_episode_animated(delay_ms=episode_delay_ms)
    gui.start()


# ============== MAIN ==============

if __name__ == "__main__":
    print("=" * 50)
    print("  Grid World 5x5 - Q-Learning")
    print("  Objetivo: ir de (0,0) ate (4,4) [*]")
    print("=" * 50)

    env = GridWorld()
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=N_ACTIONS,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    print("\nTreinando o agente...")
    steps_history = train(env, agent, n_episodes=500, max_steps=100, verbose_interval=50)

    print_policy(agent)

    print("Abrindo janela grafica: ambiente, agente em movimento e recompensa...")
    open_visualization(env, agent, episode_delay_ms=500)
    print("Fim do exemplo.")
