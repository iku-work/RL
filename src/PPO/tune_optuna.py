import optuna
#import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym 

log_dir = 'logs'

def objective(trial):
    n_steps = trial.suggest_int("n_steps", 128, 2048)
    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPO("MlpPolicy", env, n_steps=n_steps, n_epochs=n_epochs, learning_rate=learning_rate, verbose=0, tensorboard_log=log_dir)
    model.learn(total_timesteps=100000, callback=CheckpointCallback(save_freq=1000, save_path="checkpoints/"))
    
    mean_reward, _ = evaluate_policy(model, env, 5, False, False, None, None, False,False)

    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2)

#with tf.summary.create_file_writer("logs/").as_default():
#    optuna.integration.tensorboard.summary(study) #summary_target(study)