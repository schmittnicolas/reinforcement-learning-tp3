"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent

env = gym.make("Taxi-v3")
n_actions = env.action_space.n  # type: ignore


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    state, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        action = agent.get_action(state)

        next_state, reward, done, _, _ = env.step(action)

        # Train agent for state s
        # BEGIN SOLUTION
        agent.update(state, action, reward, next_state)
        total_reward += reward
        state = next_state

        if done:
            break
        # END SOLUTION

    return total_reward


def train(agent_name, agent, video_folder):
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: x % 100 == 0,
        disable_logger=True,
    )

    rewards = []
    for i in range(1001):
        rewards.append(play_and_train(env, agent))
        if i % 100 == 0:
            print(f"{agent_name}-ep{i} - mean reward: {np.mean(rewards[-100:])}")

    env.close()
    return rewards


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)

# TODO: créer des vidéos de l'agent en action

rewards = train("QLearningAgent", agent, "./videos/q_learning_agent")
assert np.mean(rewards[-100:]) > 0.0

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

# TODO: créer des vidéos de l'agent en action

rewards = train("QLearningAgentEpsScheduling", agent, "./videos/q_learning_agent_eps_scheduling")
assert np.mean(rewards[-100:]) > 0.0

#################################################
# 3. Play with SARSA
#################################################

agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = train("SARSA", agent, "./videos/sarsa")
assert np.mean(rewards[-100:]) > 0.0
