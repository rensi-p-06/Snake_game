import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =============================
# CONFIG
# =============================

WIDTH, HEIGHT = 400, 400
GRID = 20
ROWS = WIDTH // GRID

EPISODES = 1500
MAX_STEPS = 300

RENDER_INTERVAL = 1
RENDER_SPEED = 100

LR = 0.001
GAMMA = 0.9

EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

BATCH_SIZE = 64
MEMORY_SIZE = 50000

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# DQN
# =============================

class DQN(nn.Module):
    def __init__(self, input_size=11, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

# =============================
# MEMORY
# =============================

class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch):
        return random.sample(self.memory, batch)

    def __len__(self):
        return len(self.memory)

# =============================
# AGENT
# =============================

class Agent:
    def __init__(self):
        self.model = DQN().to(device)
        self.target = DQN().to(device)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory = ReplayMemory()
        self.epsilon = EPSILON

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
        a = torch.tensor(a).to(device)
        r = torch.tensor(r, dtype=torch.float32).to(device)
        ns = torch.tensor(np.array(ns), dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).to(device)

        q = self.model(s)
        next_q = self.target(ns)

        target = r + GAMMA * torch.max(next_q, 1)[0] * (1 - d)
        current = q.gather(1, a.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(current, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# =============================
# GAME
# =============================

class SnakeGame:

    def reset(self):
        self.snakes = {
            "A1": [(5,5)],
            "A2": [(5,10)],
            "B1": [(15,15)],
            "B2": [(15,10)]
        }

        self.dirs = {k: random.choice([(1,0),(-1,0),(0,1),(0,-1)]) for k in self.snakes}

        self.roles = {
            "A1": "attacker",
            "A2": "defender",
            "B1": "attacker",
            "B2": "defender"
        }

        self.spawn_food()
        return {k: self.get_state(k) for k in self.snakes}

    def spawn_food(self):
        while True:
            self.food = (random.randint(0,ROWS-1), random.randint(0,ROWS-1))
            if all(self.food not in self.snakes[k] for k in self.snakes):
                break

    def turn(self, dir, action):
        if action == 1: return (-dir[1], dir[0])
        if action == 2: return (dir[1], -dir[0])
        return dir

    def step(self, actions):

        rewards = {}
        done = False

        # update directions
        for k in self.snakes:
            self.dirs[k] = self.turn(self.dirs[k], actions[k])

        new_heads = {}
        for k in self.snakes:
            head = self.snakes[k][0]
            d = self.dirs[k]
            new_heads[k] = (head[0]+d[0], head[1]+d[1])

        dead_snakes = set()

        for k in self.snakes:

            new = new_heads[k]
            old_head = self.snakes[k][0]

            reward = -0.01

            # 💀 WALL = DEATH
            if new[0] < 0 or new[0] >= ROWS or new[1] < 0 or new[1] >= ROWS:
                reward -= 20
                dead_snakes.add(k)
                rewards[k] = reward
                continue

            self.snakes[k].insert(0, new)
            self.snakes[k].pop()

            role = self.roles[k]

            if role == "attacker":
                dist_before = abs(old_head[0]-self.food[0]) + abs(old_head[1]-self.food[1])
                dist_after  = abs(new[0]-self.food[0]) + abs(new[1]-self.food[1])

                reward += 0.3 if dist_after < dist_before else -0.3

                if new == self.food:
                    reward += 20
                    self.snakes[k].append(self.snakes[k][-1])
                    self.spawn_food()

            else:
                enemy = "B1" if k.startswith("A") else "A1"
                e = self.snakes[enemy][0]
                dist = abs(new[0]-e[0]) + abs(new[1]-e[1])
                reward += 1.0/(dist+1)

            rewards[k] = reward

        if len(dead_snakes) > 0:
            done = True

        states = {k: self.get_state(k) for k in self.snakes}
        return states, rewards, done

    def danger(self, pos):
        x,y = pos
        return int(x < 0 or x >= ROWS or y < 0 or y >= ROWS)

    def get_state(self, key):
        snake = self.snakes[key]
        dir = self.dirs[key]
        head = snake[0]

        left = (-dir[1], dir[0])
        right = (dir[1], -dir[0])

        enemy = "B1" if key.startswith("A") else "A1"
        e = self.snakes[enemy][0]

        return np.array([
            self.danger((head[0]+dir[0], head[1]+dir[1])),
            self.danger((head[0]+left[0], head[1]+left[1])),
            self.danger((head[0]+right[0], head[1]+right[1])),

            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],

            e[0] < head[0],
            e[0] > head[0],
            e[1] < head[1],
            e[1] > head[1],
        ], dtype=int)

    def draw(self):
        pygame.event.pump()
        screen.fill((0,0,0))

        colors = {
            "A1": (0,255,0),
            "A2": (0,150,0),
            "B1": (0,0,255),
            "B2": (0,0,150)
        }

        for k in self.snakes:
            for s in self.snakes[k]:
                pygame.draw.rect(screen, colors[k], (s[0]*GRID, s[1]*GRID, GRID, GRID))

        pygame.draw.circle(screen, (255,0,0),
            (self.food[0]*GRID+10, self.food[1]*GRID+10), 8)

        pygame.display.update()

# =============================
# TRAINING
# =============================

game = SnakeGame()
agents = {k: Agent() for k in ["A1","A2","B1","B2"]}

for ep in range(EPISODES):

    states = game.reset()
    render = (ep % RENDER_INTERVAL == 0)

    for step in range(MAX_STEPS):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        actions = {k: agents[k].act(states[k]) for k in agents}

        next_states, rewards, done = game.step(actions)

        agent_keys = list(agents.keys())
        for k in agents:
            agents[k].memory.push((states[k], actions[k], rewards[k], next_states[k], done))
        
        # To prevent the game from freezing, only train 1 agent per step
        agents[agent_keys[step % 4]].train()

        states = next_states

        if render:
            clock.tick(RENDER_SPEED)
            game.draw()

        if done:
            break

    for k in agents:
        agents[k].epsilon = max(MIN_EPSILON, agents[k].epsilon * EPSILON_DECAY)

    if ep % 20 == 0:
        for k in agents:
            agents[k].target.load_state_dict(agents[k].model.state_dict())

    print("Episode:", ep)

print("Training Done")

# =============================
# PLAY MODE
# =============================

states = game.reset()

while True:
    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    actions = {
        k: torch.argmax(agents[k].model(
            torch.tensor(states[k], dtype=torch.float32).to(device)
        )).item()
        for k in agents
    }

    states, _, done = game.step(actions)

    if done:
        states = game.reset()

    game.draw()