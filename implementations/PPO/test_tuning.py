import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import gym
from stable_baselines3.common.evaluation import evaluate_policy

# Define the environment
#env = gym.make("CartPole-v1")
env = gym.make("heavy_pb:driving-v0")
#env = gym.make("heavy_pb:wheel-driving-v0")

# Define the hyperparameters to be tuned
lr = [1e-4, 1e-3, 1e-2]
cliprange = [0.1, 0.2, 0.3]
ent_coef = [0.01, 0.05, 0.1]
n_steps = [320, 640, 1280]
n_epochs = [2, 4, 8]

# Define a list of hyperparameter combinations
params = []
for l in lr:
    for c in cliprange:
        for e in ent_coef:
            for n in n_steps:
                for m in n_epochs:
                    params.append([l, c, e, n, m])

# Train the model with each hyperparameter combination
best_mean_reward = float('-inf')
for i, param in enumerate(params):
    print("Training model with hyperparameters: ", param)
    model = PPO('MlpPolicy', env, learning_rate=param[0], clip_range=param[1], ent_coef=param[2], n_steps=param[3], n_epochs=param[4])
    model.learn(total_timesteps=2000)
    
    # Evaluate the model's performance
    #mean_reward, _ = model.evaluate(callback=EvalCallback(env, best_model_save_path=None, log_path=None, verbose=0))

    mean_reward, _ = evaluate_policy(model, env, 5, False, False, None, None, False,False)

    print("Mean reward: ", mean_reward)

    # Save the best model based on mean reward
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        best_model = model
        best_param = param

print("Best hyperparameters: ", best_param)
print("Best mean reward: ", best_mean_reward)

