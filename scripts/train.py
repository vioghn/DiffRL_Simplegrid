import hydra, os, wandb, yaml
from IPython.core import ultratb
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from shac.utils import hydra_utils
from hydra.utils import instantiate
from shac.utils.common import *
from shac.utils.rlgames_utils import (
    RLGPUEnvAlgoObserver,
    RLGPUEnv,
)
from gym import wrappers
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from omegaconf import OmegaConf, open_dict
import gymnasium as gym
from gym.envs.registration import register
import gym_simplegrid


try:
    from svg.train import Workspace
except:
    print_warning("SVG not installed")

# enables ipdb when script crashes
sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)
register(
    id='SimpleGrid-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200
)
register(
    id='SimpleGrid-8x8-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200,
    kwargs={'obstacle_map': '8x8'},
)
register(
    id='SimpleGrid-4x4-v0',
    entry_point='gym_simplegrid.envs:SimpleGridEnv',
    max_episode_steps=200,
    kwargs={'obstacle_map': '4x4'},
)
def register_envs(env_config):
    def create_dflex_env(**kwargs):
        # create env without grads since PPO doesn't need them
        env = instantiate(env_config.config, no_grad=True)

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1:
            env = wrappers.FrameStack(env, frames, False)

        return env

    def create_warp_env(**kwargs):
        # create env without grads since PPO doesn't need them
        env = instantiate(env_config.config, no_grad=True)

        print("num_envs = ", env.num_envs)
        print("num_actions = ", env.num_actions)
        print("num_obs = ", env.num_obs)

        frames = kwargs.pop("frames", 1)
        if frames > 1: 
            env = wrappers.FrameStack(env, frames, False)

        return env

    vecenv.register(
        "DFLEX",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "dflex",
        {
            "env_creator": lambda **kwargs: create_dflex_env(**kwargs),
            "vecenv_type": "DFLEX",
        },
    )

    vecenv.register(
        "WARP",
        lambda config_name, num_actors, **kwargs: RLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
    env_configurations.register(
        "warp",
        {
            "env_creator": lambda **kwargs: create_warp_env(**kwargs),
            "vecenv_type": "WARP",
        },
    )


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    env_name = job_config["env"]["config"]["_target_"].split(".")[-1]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    try:
        # Multirun config
        job_id = HydraConfig().get().job.num
        name = f"{alg_name}_{env_name}_sweep_{job_config['general']['seed']}"
        notes = wandb_cfg.get("notes", None)
    except:
        # Normal (singular) run config
        name = f"{alg_name}_{env_name}"
        notes = wandb_cfg["notes"]  # force user to make notes
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, "cfg")


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        create_wandb_run(cfg.wandb, cfg_full)

    # patch code to make jobs log in the correct directory when doing multirun
    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    seeding(cfg.general.seed)

    if "_target_" in cfg.alg:
        cfg.env.general.no_grad = not cfg.general.train

        algo = instantiate(cfg.alg, env_config=cfg.env.config, logdir=logdir)

        if cfg.general.checkpoint:
            algo.load(cfg.general.checkpoint)

        if cfg.general.train:
            algo.train()
        else:
            algo.run(cfg.env.player.games_num)

    elif cfg.alg.name == "ppo" or cfg.alg.name == "sac":
        # if not hydra init, then we must have PPO
        # to set up RL games we have to do a bunch of config menipulation
        # which makes it a huge mess...

        # PPO doesn't need env grads
        cfg.env.config.no_grad = True

        # first shuffle around config structure
        cfg_train = cfg_full["alg"]
        cfg_train["params"]["general"] = cfg_full["general"]
        env_name = cfg_train["params"]["config"]["env_name"]
        cfg_train["params"]["diff_env"] = cfg_full["env"]["config"]
        cfg_train["params"]["general"]["logdir"] = logdir

        # boilerplate to get rl_games working
        cfg_train["params"]["general"]["play"] = not cfg_train["params"]["general"][
            "train"
        ]

        # Now handle different env instantiation
        if env_name.split("_")[0] == "df":
            cfg_train["params"]["config"]["env_name"] = "dflex"
        elif env_name.split("_")[0] == "warp":
            cfg_train["params"]["config"]["env_name"] = "warp"
        env_name = cfg_train["params"]["diff_env"]["_target_"]
        cfg_train["params"]["diff_env"]["name"] = env_name.split(".")[-1]

        # save config
        if cfg_train["params"]["general"]["train"]:
            os.makedirs(logdir, exist_ok=True)
            yaml.dump(cfg_train, open(os.path.join(logdir, "cfg.yaml"), "w"))

        # register envs with the correct number of actors for PPO
        if cfg.alg.name == "ppo":
            cfg["env"]["config"]["num_envs"] = cfg["env"]["ppo"]["num_actors"]
        else:
            cfg["env"]["config"]["num_envs"] = cfg["env"]["sac"]["num_actors"]
        register_envs(cfg.env)

        # add observer to score keys
        if cfg_train["params"]["config"].get("score_keys"):
            algo_observer = RLGPUEnvAlgoObserver()
        else:
            algo_observer = None
        runner = Runner(algo_observer)
        runner.load(cfg_train)
        runner.reset()
        runner.run(cfg_train["params"]["general"])
    elif cfg.alg.name == "svg":
        cfg.env.config.no_grad = True
        with open_dict(cfg):
            cfg.alg.env = cfg.env.config
        w = Workspace(cfg.alg)
        w.run_epochs()
    else:
        raise NotImplementedError

    if cfg.general.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
