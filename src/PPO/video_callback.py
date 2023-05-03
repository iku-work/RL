from stable_baselines3.common.callbacks import BaseCallback
import imageio 
import os 
import gym
import numpy as np

class VideoCallback(BaseCallback):

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
        self.check_name_exsist()
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
        if(int(self.num_timesteps - self.previous_rec_timestep) > self.rec_freq):
            self.record_gif(self.model, self.env_id, self.record_len, self.video_folder, self.gif_name)
            self.previous_rec_timestep = self.num_timesteps

    def check_name_exsist(self):

        path = '{}/{}.gif'.format(self.video_folder, self.gif_name)
        i = 0
        while(os.path.isfile(path)):
            path = '{}/{}_{}.gif'.format(self.video_folder, self.gif_name, i)
            i += 1

