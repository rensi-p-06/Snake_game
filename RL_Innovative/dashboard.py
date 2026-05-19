import streamlit as st
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import plotly.graph_objects as go

st.set_page_config(page_title="IDQN Snake Dashboard", layout="wide")

# ─── Sidebar Config ───────────────────────────────────────────────────────────
st.sidebar.title("Hyperparameters")

EPISODES     = st.sidebar.slider("Episodes",        100, 5000, 2000, 100)
MAX_STEPS    = st.sidebar.slider("Max Steps",        50,  500,  300,  50)
LR           = st.sidebar.select_slider("Learning Rate", [0.0001,0.0005,0.001,0.005,0.01], value=0.001)
GAMMA        = st.sidebar.slider("Gamma",           0.5, 0.99, 0.9,  0.01)
EPSILON_DECAY= st.sidebar.slider("Epsilon Decay",   0.990, 0.999, 0.995, 0.001)
MIN_EPSILON  = st.sidebar.slider("Min Epsilon",     0.01, 0.2,  0.05, 0.01)
BATCH_SIZE   = st.sidebar.selectbox("Batch Size",   [32, 64, 128], index=1)
MEMORY_SIZE  = st.sidebar.selectbox("Memory Size",  [10000, 50000, 100000], index=1)
TARGET_UPDATE= st.sidebar.slider("Target Update Interval", 5, 100, 20, 5)
ROWS         = st.sidebar.slider("Grid Size (NxN)", 10, 30, 20, 1)
RENDER_LIVE  = st.sidebar.checkbox("Render Live (Slows Training)", value=True)

EPSILON = 1.0
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model ────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
    def push(self, t):
        self.memory.append(t)
    def sample(self, n):
        return random.sample(self.memory, n)
    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self):
        self.model  = DQN().to(device)
        self.target = DQN().to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.memory  = ReplayMemory()
        self.epsilon = EPSILON

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        s = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            return torch.argmax(self.model(s)).item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states      = torch.tensor(np.array(states),      dtype=torch.float32).to(device)
        actions     = torch.tensor(actions).to(device)
        rewards     = torch.tensor(rewards,               dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones       = torch.tensor(dones,                 dtype=torch.float32).to(device)
        q_values    = self.model(states)
        next_q      = self.target(next_states)
        target      = rewards + GAMMA * torch.max(next_q, 1)[0] * (1 - dones)
        current     = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss        = nn.MSELoss()(current, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# ─── Environment ──────────────────────────────────────────────────────────────
class SnakeGame:
    def reset(self):
        self.snakeA = [(random.randint(3, ROWS-3), random.randint(3, ROWS-3))]
        self.snakeB = [(random.randint(3, ROWS-3), random.randint(3, ROWS-3))]
        self.dirA   = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        self.dirB   = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        self.scoreA = 0
        self.scoreB = 0
        self.spawn_food()
        return self.get_state_A(), self.get_state_B()

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, ROWS-1), random.randint(0, ROWS-1))
            if self.food not in self.snakeA and self.food not in self.snakeB:
                break

    def turn(self, d, a):
        if a == 1: return (-d[1],  d[0])
        if a == 2: return ( d[1], -d[0])
        return d

    def step(self, aA, aB):
        dA_b = abs(self.snakeA[0][0]-self.food[0]) + abs(self.snakeA[0][1]-self.food[1])
        dB_b = abs(self.snakeB[0][0]-self.food[0]) + abs(self.snakeB[0][1]-self.food[1])
        self.dirA = self.turn(self.dirA, aA)
        self.dirB = self.turn(self.dirB, aB)
        nhA = (self.snakeA[0][0]+self.dirA[0], self.snakeA[0][1]+self.dirA[1])
        nhB = (self.snakeB[0][0]+self.dirB[0], self.snakeB[0][1]+self.dirB[1])
        rA, rB, done = -0.01, -0.01, False
        if nhA[0]<0 or nhA[0]>=ROWS or nhA[1]<0 or nhA[1]>=ROWS:
            rA -= 1; hA = self.snakeA[0]
        else:
            hA = nhA; self.snakeA.insert(0, hA); self.snakeA.pop()
        if nhB[0]<0 or nhB[0]>=ROWS or nhB[1]<0 or nhB[1]>=ROWS:
            rB -= 1; hB = self.snakeB[0]
        else:
            hB = nhB; self.snakeB.insert(0, hB); self.snakeB.pop()
        dA_a = abs(hA[0]-self.food[0]) + abs(hA[1]-self.food[1])
        dB_a = abs(hB[0]-self.food[0]) + abs(hB[1]-self.food[1])
        rA += 0.3 if dA_a < dA_b else -0.3
        rB += 0.3 if dB_a < dB_b else -0.3
        if hA == self.food:
            rA += 20; self.scoreA += 1
            self.snakeA.append(self.snakeA[-1]); self.spawn_food()
        if hB == self.food:
            rB += 20; self.scoreB += 1
            self.snakeB.append(self.snakeB[-1]); self.spawn_food()
        return self.get_state_A(), self.get_state_B(), rA, rB, done

    def danger(self, snake, d):
        x, y = snake[0][0]+d[0], snake[0][1]+d[1]
        return 1 if (x<0 or x>=ROWS or y<0 or y>=ROWS) else 0

    def get_state(self, snake, d, enemy):
        left  = (-d[1],  d[0])
        right = ( d[1], -d[0])
        eh    = enemy[0]
        return np.array([
            self.danger(snake, d), self.danger(snake, left), self.danger(snake, right),
            self.food[0]<snake[0][0], self.food[0]>snake[0][0],
            self.food[1]<snake[0][1], self.food[1]>snake[0][1],
            eh[0]<snake[0][0], eh[0]>snake[0][0],
            eh[1]<snake[0][1], eh[1]>snake[0][1]
        ], dtype=int)

    def get_state_A(self): return self.get_state(self.snakeA, self.dirA, self.snakeB)
    def get_state_B(self): return self.get_state(self.snakeB, self.dirB, self.snakeA)

    def render_plotly(self):
        """Render the game board as a crisp Plotly figure using shapes."""
        CELL = 20  # px per cell
        SIZE = ROWS * CELL

        shapes = []

        def cell(r, c, color):
            shapes.append(dict(
                type="rect",
                x0=c * CELL, y0=SIZE - (r+1)*CELL,
                x1=(c+1)*CELL, y1=SIZE - r*CELL,
                fillcolor=color, line=dict(width=0)
            ))

        # Grid lines (subtle)
        for i in range(ROWS+1):
            shapes.append(dict(type="line", x0=i*CELL, y0=0, x1=i*CELL, y1=SIZE,
                               line=dict(color="#1a1a2e", width=1)))
            shapes.append(dict(type="line", x0=0, y0=i*CELL, x1=SIZE, y1=i*CELL,
                               line=dict(color="#1a1a2e", width=1)))

        # Snake A (green)
        for idx, s in enumerate(self.snakeA):
            color = "#00e600" if idx == 0 else "#009900"
            cell(s[1], s[0], color)

        # Snake B (blue)
        for idx, s in enumerate(self.snakeB):
            color = "#3399ff" if idx == 0 else "#005cb8"
            cell(s[1], s[0], color)

        # Food (red circle via scatter)
        fx = self.food[0] * CELL + CELL // 2
        fy = SIZE - self.food[1] * CELL - CELL // 2

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[fx], y=[fy],
            mode="markers",
            marker=dict(color="#ff3232", size=CELL * 0.7, symbol="circle"),
            showlegend=False
        ))

        fig.update_layout(
            shapes=shapes,
            xaxis=dict(range=[0, SIZE], showgrid=False, zeroline=False,
                       showticklabels=False, fixedrange=True),
            yaxis=dict(range=[0, SIZE], showgrid=False, zeroline=False,
                       showticklabels=False, fixedrange=True, scaleanchor="x"),
            plot_bgcolor="#0d0d1a",
            paper_bgcolor="#0d0d1a",
            margin=dict(l=0, r=0, t=0, b=0),
            height=SIZE, width=SIZE,
        )
        return fig

# ─── UI Layout ────────────────────────────────────────────────────────────────
st.title("🐍 IDQN Multi-Agent Snake — Training Dashboard")
st.caption(f"Device: `{device}` | Grid: {ROWS}×{ROWS}")

col_start, col_stop, col_reset = st.columns([1,1,4])
start_btn = col_start.button("▶ Start Training", type="primary")
stop_btn  = col_stop.button("⏹ Stop")

if "running"      not in st.session_state: st.session_state.running      = False
if "scores_a"     not in st.session_state: st.session_state.scores_a     = []
if "scores_b"     not in st.session_state: st.session_state.scores_b     = []
if "epsilons"     not in st.session_state: st.session_state.epsilons     = []
if "losses_a"     not in st.session_state: st.session_state.losses_a     = []
if "losses_b"     not in st.session_state: st.session_state.losses_b     = []
if "episode"      not in st.session_state: st.session_state.episode      = 0
if "agentA"       not in st.session_state: st.session_state.agentA       = None
if "agentB"       not in st.session_state: st.session_state.agentB       = None
if "game"         not in st.session_state: st.session_state.game         = None

if start_btn:
    st.session_state.running  = True
    st.session_state.scores_a = []
    st.session_state.scores_b = []
    st.session_state.epsilons = []
    st.session_state.losses_a = []
    st.session_state.losses_b = []
    st.session_state.episode  = 0
    st.session_state.agentA   = Agent()
    st.session_state.agentB   = Agent()
    st.session_state.game     = SnakeGame()

if stop_btn:
    st.session_state.running = False

# ─── Metrics Row ──────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
metric_episode  = m1.empty()
metric_scoreA   = m2.empty()
metric_scoreB   = m3.empty()
metric_epsilon  = m4.empty()
metric_device   = m5.empty()

metric_device.metric("Device", str(device).upper())

# ─── Charts ───────────────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    score_chart = st.empty()
with chart_col2:
    loss_chart  = st.empty()

# ─── Live Game Board (full width, crisp) ──────────────────────────────────────
st.markdown("**Live Game Board**")
st.markdown(
    "<span style='color:#00e600'>■</span> Agent A Head &nbsp;"
    "<span style='color:#009900'>■</span> Agent A Body &nbsp;&nbsp;"
    "<span style='color:#3399ff'>■</span> Agent B Head &nbsp;"
    "<span style='color:#005cb8'>■</span> Agent B Body &nbsp;&nbsp;"
    "<span style='color:#ff3232'>●</span> Food",
    unsafe_allow_html=True
)
game_view = st.empty()

# ─── Training Loop ────────────────────────────────────────────────────────────
CHART_SMOOTH = 10   # rolling average window

def smooth(arr, w):
    if len(arr) < w: return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid').tolist()

def render_score_chart(sa, sb):
    fig = go.Figure()
    eps = list(range(1, len(sa)+1))
    fig.add_trace(go.Scatter(x=eps, y=sa, name="Agent A", line=dict(color="#00c800", width=1), opacity=0.35))
    fig.add_trace(go.Scatter(x=eps, y=sb, name="Agent B", line=dict(color="#0064c8", width=1), opacity=0.35))
    if len(sa) >= CHART_SMOOTH:
        fig.add_trace(go.Scatter(x=list(range(CHART_SMOOTH, len(sa)+1)), y=smooth(sa, CHART_SMOOTH),
                                 name=f"A (avg{CHART_SMOOTH})", line=dict(color="#00ff00", width=2)))
        fig.add_trace(go.Scatter(x=list(range(CHART_SMOOTH, len(sb)+1)), y=smooth(sb, CHART_SMOOTH),
                                 name=f"B (avg{CHART_SMOOTH})", line=dict(color="#00aaff", width=2)))
    fig.update_layout(title="Score per Episode", xaxis_title="Episode", yaxis_title="Score",
                      height=280, margin=dict(l=40,r=20,t=40,b=30),
                      legend=dict(orientation="h", y=1.15), template="plotly_dark")
    return fig

def render_loss_chart(la, lb, epsilons):
    fig = go.Figure()
    eps = list(range(1, len(la)+1))
    fig.add_trace(go.Scatter(x=eps, y=la, name="Loss A", line=dict(color="#ffaa00", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=eps, y=lb, name="Loss B", line=dict(color="#ff5500", width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=list(range(1, len(epsilons)+1)), y=epsilons,
                             name="Epsilon", line=dict(color="#cc88ff", width=2, dash="dot"),
                             yaxis="y2"))
    fig.update_layout(
        title="Loss & Epsilon", xaxis_title="Episode", height=280,
        margin=dict(l=40,r=60,t=40,b=30), template="plotly_dark",
        legend=dict(orientation="h", y=1.15),
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Epsilon", overlaying="y", side="right", range=[0,1])
    )
    return fig

if st.session_state.running and st.session_state.agentA is not None:
    agentA = st.session_state.agentA
    agentB = st.session_state.agentB
    game   = st.session_state.game

    start_ep = st.session_state.episode

    for episode in range(start_ep, EPISODES):
        if not st.session_state.running:
            break

        stateA, stateB = game.reset()
        ep_lossA, ep_lossB = [], []

        for step in range(MAX_STEPS):
            aA = agentA.act(stateA)
            aB = agentB.act(stateB)
            nA, nB, rA, rB, done = game.step(aA, aB)
            agentA.memory.push((stateA, aA, rA, nA, done))
            agentB.memory.push((stateB, aB, rB, nB, done))
            lA = agentA.train_step()
            lB = agentB.train_step()
            if lA is not None: ep_lossA.append(lA)
            if lB is not None: ep_lossB.append(lB)
            stateA, stateB = nA, nB

            if RENDER_LIVE:
                game_view.plotly_chart(game.render_plotly(), use_container_width=True, key=f"live_{episode}_{step}")
                # Small sleep to simulate pygame-like frames
                time.sleep(0.05)

        agentA.epsilon = max(MIN_EPSILON, agentA.epsilon * EPSILON_DECAY)
        agentB.epsilon = max(MIN_EPSILON, agentB.epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            agentA.target.load_state_dict(agentA.model.state_dict())
            agentB.target.load_state_dict(agentB.model.state_dict())

        st.session_state.scores_a.append(game.scoreA)
        st.session_state.scores_b.append(game.scoreB)
        st.session_state.epsilons.append(agentA.epsilon)
        st.session_state.losses_a.append(np.mean(ep_lossA) if ep_lossA else 0)
        st.session_state.losses_b.append(np.mean(ep_lossB) if ep_lossB else 0)
        st.session_state.episode = episode + 1

        # Update UI every 5 episodes to keep it snappy
        if episode % 5 == 0 or episode == EPISODES - 1:
            sa = st.session_state.scores_a
            sb = st.session_state.scores_b

            metric_episode.metric("Episode", f"{episode+1} / {EPISODES}")
            metric_scoreA.metric("Agent A Score", game.scoreA,
                                 delta=int(game.scoreA - (sa[-2] if len(sa)>1 else 0)))
            metric_scoreB.metric("Agent B Score", game.scoreB,
                                 delta=int(game.scoreB - (sb[-2] if len(sb)>1 else 0)))
            metric_epsilon.metric("Epsilon", f"{agentA.epsilon:.3f}")

            score_chart.plotly_chart(render_score_chart(sa, sb), use_container_width=True, key=f"score_{episode}")
            loss_chart.plotly_chart(
                render_loss_chart(st.session_state.losses_a, st.session_state.losses_b, st.session_state.epsilons),
                use_container_width=True,
                key=f"loss_{episode}"
            )
            if not RENDER_LIVE:
                game_view.plotly_chart(game.render_plotly(), use_container_width=True, key=f"board_{episode}")

    st.session_state.running = False
    st.success(f"Training complete — {EPISODES} episodes finished.")

elif not st.session_state.running and st.session_state.scores_a:
    # Show final charts after training stops
    sa = st.session_state.scores_a
    sb = st.session_state.scores_b
    metric_episode.metric("Episode", f"{st.session_state.episode} / {EPISODES}")
    metric_scoreA.metric("Agent A Best", max(sa))
    metric_scoreB.metric("Agent B Best", max(sb))
    metric_epsilon.metric("Final Epsilon", f"{st.session_state.epsilons[-1]:.3f}")
    score_chart.plotly_chart(render_score_chart(sa, sb), use_container_width=True, key="score_final")
    loss_chart.plotly_chart(
        render_loss_chart(st.session_state.losses_a, st.session_state.losses_b, st.session_state.epsilons),
        use_container_width=True,
        key="loss_final"
    )
else:
    metric_episode.metric("Episode", "—")
    metric_scoreA.metric("Agent A Score", "—")
    metric_scoreB.metric("Agent B Score", "—")
    metric_epsilon.metric("Epsilon", "—")
    score_chart.info("Press ▶ Start Training to begin.")
