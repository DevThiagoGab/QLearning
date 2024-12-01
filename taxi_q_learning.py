import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3", render_mode="human")  
n_states = env.observation_space.n
n_actions = env.action_space.n


if not hasattr(np, "bool8"):
    np.bool8 = bool  


alpha = 0.1  
gamma = 0.99  
epsilon = 1.0  
epsilon_decay = 0.995  
epsilon_min = 0.1  
episodes = 5000  
max_steps = 100  


Q = np.zeros((n_states, n_actions))


rewards = []
epsilon_values = []


for episode in range(episodes):
    state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()  # Corrige o retorno
    state = int(state)  # Garante que seja um inteiro
    total_reward = 0

    for step in range(max_steps):
        # Seleciona a ação (estratégia epsilon-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(Q[state, :])  # Exploração (melhor ação)

        action = int(action)  # Garante que seja um inteiro

       
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = int(next_state)  # Garante que o próximo estado seja inteiro
        done = bool(terminated or truncated)

        # Atualiza a Tabela Q
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        
        state = next_state
        total_reward += reward

        
        if done:
            break

    
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

   
    rewards.append(total_reward)
    epsilon_values.append(epsilon)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel("Episódios")
plt.ylabel("Recompensa acumulada")
plt.title("Recompensa por episódio")


plt.subplot(1, 2, 2)
plt.plot(epsilon_values)
plt.xlabel("Episódios")
plt.ylabel("Valor de Epsilon")
plt.title("Decaimento de Epsilon")

plt.tight_layout()
plt.show()

state = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
state = int(state)
done = False
total_reward = 0

print("\nAvaliando o agente treinado...\n")
while not done:
    action = int(np.argmax(Q[state, :]))  # Melhor ação com base na Tabela Q
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = int(next_state)
    done = bool(terminated or truncated)
    total_reward += reward
    state = next_state

    
    env.render()

print(f"\nRecompensa total no episódio de teste: {total_reward}")
env.close()