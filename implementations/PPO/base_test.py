import gym 
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import os 

from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import TensorBoardOutputFormat
import imageio 


class CustomCallback(BaseCallback):

    def __init__(self, video_folder: str, verbose: int = 0,  env_id: str = '', record_len: int = 500, gif_name: str = '', rec_freq: int = 10000):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.video_folder = video_folder
        self.env_id = env_id
        self.record_len = record_len
        self.gif_name = gif_name
        self.previous_rec_timestep = 0
        self.rec_freq = rec_freq

    def record_gif(self, model, env_id, record_len, video_folder, gif_name):

        gif_name = '{}_{}'.format(gif_name, self.num_timesteps)
        env = gym.make(env_id)  
    
        images = []
        obs = env.reset()
        img = env.render(mode="rgb_array")
        for i in range(record_len):

            images.append(img)
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rew, done ,_ = env.step(action)
            img = env.render(mode="rgb_array")

            if done:
                obs = env.reset()

        isExist = os.path.exists(video_folder)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(video_folder)

        env_name = env_id.split(':')[-1]
        imageio.mimsave(video_folder + "/result_{}_{}.gif".format(gif_name, env_name), [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
        env.close()
    
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if(self.num_timesteps - self.previous_rec_timestep > self.rec_freq):
            self.record_gif(self.model, self.env_id, self.record_len, self.video_folder, self.gif_name)



save_dir = 'models/PPO'
log_dir = 'logs'

if(os.name != 'posix'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#if not os.path.exists(save_dir):
#    os.makedirs(save_dir)  

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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

if __name__ == '__main__':

    env_name = 'forwarder-v0'
    env_id = "heavy_pb:{}".format(env_name) 
    num_cpu = 3  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    #env = gym.make(env_id)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(env ,
                                best_model_save_path='../../models',
                                log_path=log_dir,
                                eval_freq=12000,
                                n_eval_episodes=10,
                                deterministic=True,
                                render=False,
                                callback_on_new_best=None)

    #ent_coefs = [.01, .05, .1, .5]
    frame_skips = [10, 30, 60, 80, 120]


    for fs in frame_skips:
        video_folder = "logs/videos/{}_{}/".format('fs', fs) 
        customCallback = CustomCallback(video_folder=video_folder, 
                                        env_id=env_id, 
                                        gif_name='fs_{}'.format(fs),
                                        rec_freq=1e6
                                        )

        env.env_method('set_frame_skip', fs)

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, n_steps=4096*num_cpu)

        #model = PPO('MlpPolicy', env, learning_rate=param[0], clip_range=param[1], ent_coef=param[2], n_steps=param[3], n_epochs=param[4])
        model.learn(total_timesteps=350000, 
                    tb_log_name='ppo_{}_{}_{}'.format(env_name, 'fs', fs), 
                    callback=[eval_callback, customCallback]
                    )
        model.save(save_dir + 'control_{}_fs{}'.format(env_id, str(fs)))
        
