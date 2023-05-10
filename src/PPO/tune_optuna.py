import optuna
#import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import gym 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from video_callback import VideoCallback

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, filename=None)
        return env
    set_random_seed(seed)
    return _init

log_dir = 'logs'
env_name = 'forwarder-v0'
env_id = "heavy_pb:{}".format(env_name) 
num_cpu = 3  # Number of processes to use
# Create the vectorized environment


def objective(trial):
    n_steps = trial.suggest_int("n_steps", 128, 2048, 128)
    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    ent_coef = trial.suggest_float("learning_rate", 0, 0.01, 0.002)
    

    model = PPO("MlpPolicy", env, n_steps=n_steps, n_epochs=n_epochs, learning_rate=learning_rate, ent_coef=ent_coef, verbose=0, tensorboard_log=log_dir)
    video_folder = "logs/videos/{}/".format(env_name) 
    customCallback = VideoCallback(video_folder=video_folder, 
                                    env_id=env_id, 
                                    gif_name='{}'.format(env_name),
                                    rec_freq=2000
                                    )
    model.learn(total_timesteps=6000, callback=customCallback)
    
    mean_reward, _ = evaluate_policy(model, env, 5, False, False, None, None, False,False)

    return mean_reward

if __name__ == '__main__':

    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

#with tf.summary.create_file_writer("logs/").as_default():
#    optuna.integration.tensorboard.summary(study) #summary_target(study)