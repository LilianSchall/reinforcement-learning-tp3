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
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="rgb_array")
# env = RecordVideo(
#     env,
#     video_folder="videos",
#     name_prefix="training",
#     episode_trigger=lambda x: x % 1 == 0
# )
# env = RecordEpisodeStatistics(env)
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

# You can edit these hyperparameters!
agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        total_reward += r
        agent.update(s, a, r, next_s)
        s = next_s
        if done:
            break
        # END SOLUTION

    return total_reward

def grid_search(env: gym.Env, agent_class):
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    epsilons = [0.01, 0.05, 0.1, 0.2]
    gammas = [0.9, 0.95, 0.99, 0.999]

    all_rewards = np.full((len(learning_rates), len(epsilons), len(gammas)), -np.inf)
    
    for lr_id, lr in enumerate(learning_rates):
        for e_id, e in enumerate(epsilons):
            for g_id, g in enumerate(gammas):
                agent = agent_class(
                    learning_rate=lr, epsilon=e, gamma=g, legal_actions=list(range(n_actions))
                )
                rewards = []
                for i in range(1000):
                    rewards.append(play_and_train(env, agent))
                mean_reward = np.mean(rewards[-100:])
                print(f"lr={lr}, epsilon={e}, gamma={g} -> mean reward = {mean_reward}")
                all_rewards[lr_id,e_id,g_id] = mean_reward
                
    fig, axs = plt.subplots(1, len(learning_rates), figsize=(18, 6))

    for lr_id in range(len(learning_rates)):
        im = axs[lr_id].imshow(all_rewards[lr_id])
        axs[lr_id].set_xticks(np.arange(len(gammas)), labels=gammas)
        axs[lr_id].set_yticks(np.arange(len(epsilons)), labels=epsilons)
        axs[lr_id].set_title(f"Grid search with learning_rate={learning_rates[lr_id]}")
        axs[lr_id].set_xlabel("Gamma") 
        axs[lr_id].set_ylabel("Epsilon")
    
        for i in range(len(epsilons)):
            for j in range(len(gammas)):
                axs[lr_id].text(j, i, f"{all_rewards[lr_id][i, j]:.2f}", ha='center', va='center', color='white')

    plt.savefig(f"grid_search_{agent_class.__name__}.png", bbox_inches='tight')
    
    best_lr_id, best_e_id, best_g_id = np.unravel_index(np.argmax(all_rewards), all_rewards.shape)
    best_lr, best_e, best_g = learning_rates[best_lr_id], epsilons[best_e_id], gammas[best_g_id]
    print(f"Best hyperparameters: lr={best_lr}, epsilon={best_e}, gamma={best_g}")

    return best_lr, best_e, best_g

best_lr, best_e, best_g = grid_search(env, QLearningAgent)
agent = QLearningAgent(
    learning_rate=best_lr, epsilon=best_e, gamma=best_g, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

#assert np.mean(rewards[-100:]) > 0.0

# créer des vidéos de l'agent en action
env = gym.make("Taxi-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos",
    name_prefix="qlearning",
    episode_trigger=lambda x: x % 1 == 0
)
env = RecordEpisodeStatistics(env)
print("Reward for testing: " + str(play_and_train(env, agent)))
env.close()

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

best_lr, best_e, best_g = grid_search(env, QLearningAgentEpsScheduling)
agent = QLearningAgentEpsScheduling(
    learning_rate=best_lr, epsilon=best_e, gamma=best_g, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

#assert np.mean(rewards[-100:]) > 0.0

# créer des vidéos de l'agent en action
env = gym.make("Taxi-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos",
    name_prefix="qlearning_eps_scheduling",
    episode_trigger=lambda x: x % 1 == 0
)
env = RecordEpisodeStatistics(env)
print("Reward for testing: " + str(play_and_train(env, agent)))
env.close()


####################
# 3. Play with SARSA
####################

agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

env = gym.make("Taxi-v3", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="videos",
    name_prefix="sarsa",
    episode_trigger=lambda x: x % 1 == 0
)
env = RecordEpisodeStatistics(env)
print("Reward for testing: " + str(play_and_train(env, agent)))
env.close()
