import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =============================
# Environment Parameters
# =============================

WIDTH = 400
HEIGHT = 400
GRID = 20
ROWS = WIDTH // GRID

EPISODES = 2000
MAX_STEPS = 300

RENDER_INTERVAL = 1
RENDER_SPEED = 15

LR = 0.001
GAMMA = 0.9

EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

BATCH_SIZE = 64
MEMORY_SIZE = 50000

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("IDQN Multi-Agent Snake")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial",18)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# Neural Network
# =============================

class DQN(nn.Module):

    def __init__(self,input_size,output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,output_size)
        )

    def forward(self,x):
        return self.net(x)


# =============================
# Replay Memory
# =============================

class ReplayMemory:

    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def push(self,transition):
        self.memory.append(transition)

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


# =============================
# Snake Environment
# =============================

class SnakeGame:

    def reset(self):

        self.snakeA = [(random.randint(3,ROWS-3),random.randint(3,ROWS-3))]
        self.snakeB = [(random.randint(3,ROWS-3),random.randint(3,ROWS-3))]

        self.dirA = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        self.dirB = random.choice([(1,0),(-1,0),(0,1),(0,-1)])

        self.scoreA = 0
        self.scoreB = 0

        self.spawn_food()

        return self.get_state_A(), self.get_state_B()


    def spawn_food(self):

        while True:

            self.food = (random.randint(0,ROWS-1),random.randint(0,ROWS-1))

            if self.food not in self.snakeA and self.food not in self.snakeB:
                break


    def turn(self,dir,action):

        if action == 1:
            return (-dir[1],dir[0])

        if action == 2:
            return (dir[1],-dir[0])

        return dir


    def step(self,actionA,actionB):

        distA_before = abs(self.snakeA[0][0]-self.food[0]) + abs(self.snakeA[0][1]-self.food[1])
        distB_before = abs(self.snakeB[0][0]-self.food[0]) + abs(self.snakeB[0][1]-self.food[1])

        self.dirA = self.turn(self.dirA,actionA)
        self.dirB = self.turn(self.dirB,actionB)

        new_headA = (self.snakeA[0][0]+self.dirA[0], self.snakeA[0][1]+self.dirA[1])
        new_headB = (self.snakeB[0][0]+self.dirB[0], self.snakeB[0][1]+self.dirB[1])

        rewardA = -0.01
        rewardB = -0.01

        done = False

        # Wall collision
        if new_headA[0] < 0 or new_headA[0] >= ROWS or new_headA[1] < 0 or new_headA[1] >= ROWS:
            rewardA -= 1
            headA = self.snakeA[0]
        else:
            headA = new_headA
            self.snakeA.insert(0,headA)
            self.snakeA.pop()

        if new_headB[0] < 0 or new_headB[0] >= ROWS or new_headB[1] < 0 or new_headB[1] >= ROWS:
            rewardB -= 1
            headB = self.snakeB[0]
        else:
            headB = new_headB
            self.snakeB.insert(0,headB)
            self.snakeB.pop()

        distA_after = abs(headA[0]-self.food[0]) + abs(headA[1]-self.food[1])
        distB_after = abs(headB[0]-self.food[0]) + abs(headB[1]-self.food[1])

        rewardA += 0.3 if distA_after < distA_before else -0.3
        rewardB += 0.3 if distB_after < distB_before else -0.3

        if headA == self.food:
            rewardA += 20
            self.scoreA += 1
            self.snakeA.append(self.snakeA[-1])
            self.spawn_food()

        if headB == self.food:
            rewardB += 20
            self.scoreB += 1
            self.snakeB.append(self.snakeB[-1])
            self.spawn_food()

        return self.get_state_A(), self.get_state_B(), rewardA, rewardB, done


    def danger(self,snake,dir):

        x = snake[0][0] + dir[0]
        y = snake[0][1] + dir[1]

        if x < 0 or x >= ROWS or y < 0 or y >= ROWS:
            return 1

        return 0


    def get_state(self,snake,dir,enemy):

        left = (-dir[1],dir[0])
        right = (dir[1],-dir[0])

        enemy_head = enemy[0]

        state = [

            self.danger(snake,dir),
            self.danger(snake,left),
            self.danger(snake,right),

            self.food[0] < snake[0][0],
            self.food[0] > snake[0][0],
            self.food[1] < snake[0][1],
            self.food[1] > snake[0][1],

            enemy_head[0] < snake[0][0],
            enemy_head[0] > snake[0][0],
            enemy_head[1] < snake[0][1],
            enemy_head[1] > snake[0][1]
        ]

        return np.array(state,dtype=int)


    def get_state_A(self):
        return self.get_state(self.snakeA,self.dirA,self.snakeB)

    def get_state_B(self):
        return self.get_state(self.snakeB,self.dirB,self.snakeA)


    def draw(self,episode,steps,epsilon):

        screen.fill((0,0,0))

        for s in self.snakeA:
            pygame.draw.rect(screen,(0,200,0),(s[0]*GRID,s[1]*GRID,GRID,GRID))

        for s in self.snakeB:
            pygame.draw.rect(screen,(0,0,200),(s[0]*GRID,s[1]*GRID,GRID,GRID))

        pygame.draw.circle(
            screen,
            (255,50,50),
            (self.food[0]*GRID + GRID//2, self.food[1]*GRID + GRID//2),
            GRID//2
        )

        text = font.render(f"Episode: {episode}",True,(255,255,255))
        screen.blit(text,(10,10))

        text2 = font.render(f"Steps: {steps}",True,(255,255,255))
        screen.blit(text2,(10,30))

        text3 = font.render(f"Epsilon: {epsilon:.2f}",True,(255,255,255))
        screen.blit(text3,(10,50))

        text4 = font.render(f"ScoreA: {self.scoreA}",True,(0,255,0))
        screen.blit(text4,(10,70))

        text5 = font.render(f"ScoreB: {self.scoreB}",True,(0,150,255))
        screen.blit(text5,(10,90))

        pygame.display.update()


# =============================
# Agent
# =============================

class Agent:

    def __init__(self):

        self.model = DQN(11,3).to(device)
        self.target = DQN(11,3).to(device)

        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(),lr=LR)

        self.memory = ReplayMemory()

        self.epsilon = EPSILON


    def act(self,state):

        if random.random() < self.epsilon:
            return random.randint(0,2)

        state = torch.tensor(state,dtype=torch.float32).to(device)

        with torch.no_grad():
            q = self.model(state)

        return torch.argmax(q).item()


    def train(self):

        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)

        states,actions,rewards,next_states,dones = zip(*batch)

        states = torch.tensor(np.array(states),dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards,dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32).to(device)
        dones = torch.tensor(dones,dtype=torch.float32).to(device)

        q_values = self.model(states)
        next_q = self.target(next_states)

        target = rewards + GAMMA * torch.max(next_q,1)[0] * (1-dones)

        current = q_values.gather(1,actions.unsqueeze(1)).squeeze()

        loss = nn.MSELoss()(current,target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# =============================
# Training
# =============================

game = SnakeGame()

agentA = Agent()
agentB = Agent()

for episode in range(EPISODES):

    stateA,stateB = game.reset()
    steps = 0

    render = (episode % RENDER_INTERVAL == 0)

    while steps < MAX_STEPS:

        steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        actionA = agentA.act(stateA)
        actionB = agentB.act(stateB)

        nextA,nextB,rA,rB,done = game.step(actionA,actionB)

        agentA.memory.push((stateA,actionA,rA,nextA,done))
        agentB.memory.push((stateB,actionB,rB,nextB,done))

        agentA.train()
        agentB.train()

        stateA = nextA
        stateB = nextB

        if render:
            clock.tick(RENDER_SPEED)
            game.draw(episode,steps,agentA.epsilon)

    agentA.epsilon = max(MIN_EPSILON,agentA.epsilon*EPSILON_DECAY)
    agentB.epsilon = max(MIN_EPSILON,agentB.epsilon*EPSILON_DECAY)

    if episode % 20 == 0:
        agentA.target.load_state_dict(agentA.model.state_dict())
        agentB.target.load_state_dict(agentB.model.state_dict())

    print("Episode:",episode,"ScoreA:",game.scoreA,"ScoreB:",game.scoreB)


print("Training Finished")


# =============================
# Final Evaluation Mode
# =============================

stateA,stateB = game.reset()

while True:

    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    actionA = torch.argmax(agentA.model(torch.tensor(stateA,dtype=torch.float32).to(device))).item()
    actionB = torch.argmax(agentB.model(torch.tensor(stateB,dtype=torch.float32).to(device))).item()

    nextA,nextB,rA,rB,done = game.step(actionA,actionB)

    stateA = nextA
    stateB = nextB

    if done:
        stateA,stateB = game.reset()

    game.draw("FINAL",0,0)