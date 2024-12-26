import argparse
import datetime
import os
import random
import importlib
from copy import deepcopy

import gym
import d4rl
import neorl

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian, ValNet
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer_gpu import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger, load_neorl_dataset
from static_fns.termination_fns import get_termination_fn
from config import loaded_args

torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="neorl")
    parser.add_argument("--algo-name", type=str, default="mopo")
    parser.add_argument("--task", type=str, default="walker2d-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=1)

    # dynamics model's arguments
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)

    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-model", action='store_true')
    
    known_args, _ = parser.parse_known_args()
    # import configs
    domain = known_args.domain
    if domain == "gym":
        task = known_args.task.split('-')[0]
        # import_path = f"static_fns.{task}"
        # config_path = f"config.{domain}.{task}"
        # config = importlib.import_module(config_path).default_config
        task_config = loaded_args[task]
    elif domain == "neorl":
        task_config = loaded_args[known_args.task]
    
    for arg_key, arg_value in task_config.items():
        parser.add_argument(f'--{arg_key}', default=arg_value, type=type(arg_value))

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    assert args.domain in ["gym", "neorl"]
    if args.domain == "neorl":
        task, version, data_type = tuple(args.task.split("-"))
        env = neorl.make(task+'-'+version)
        dataset = load_neorl_dataset(env, data_type)
    else:
        env = gym.make(args.task)
        dataset = d4rl.qlearning_dataset(env)
    
    domain = args.domain
    if domain == "gym":
        task = args.task.split('-')[0]
        import_path = f"static_fns.{task}"
        static_fns = importlib.import_module(import_path).StaticFns
        # config_path = f"config.{domain}.{task}"
        # config = importlib.import_module(config_path).default_config
    elif domain == "neorl":
        static_fns = get_termination_fn(task=args.task)   
    
    if args.norm_reward:
        r_mean, r_std = dataset["rewards"].mean(), dataset["rewards"].std()
        dataset["rewards"] = (dataset["rewards"] - r_mean) / (r_std + 1e-3)
        
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.device != "cpu":
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # env.seed(args.seed)

    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.beta}_{args.lbd}_{args.phi}_{args.real_ratio}'
    log_path = os.path.join(args.logdir, args.task, log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = Logger(writer=writer,log_path=log_path)

    Devid = 0 if args.device == 'cuda' else -1
    set_device_and_logger(Devid,logger)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    value_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    value_backbone_1 = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    true_valnet = ValNet(value_backbone, args.device)
    true_valnet_1 = ValNet(value_backbone_1, args.device)
    model_valnet = ValNet(value_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    val_optim = torch.optim.Adam(true_valnet.parameters(), lr=args.value_lr)
    val_optim_1 = torch.optim.Adam(true_valnet_1.parameters(), lr=args.value_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        true_valnet,
        true_valnet_1,
        actor_optim,
        critic1_optim,
        critic2_optim,
        val_optim,
        val_optim_1,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        lbd=args.lbd,
        beta=args.beta,
        device=args.device
    )

    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     true_valnet=deepcopy(true_valnet),
                                     model_valnet=model_valnet,
                                     lr=args.dynamics_lr,
                                     penalty_coeff=args.reward_penalty_coef,
                                     pretrain=args.load_model,
                                     phi=args.phi,
                                     **args.transition_params
                                     )

    # create buffer
    offline_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=torch.float32,
        action_dim=args.action_dim,
        action_dtype=torch.float32,
        device=args.device
    )
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=torch.float32,
        action_dim=args.action_dim,
        action_dtype=torch.float32,
        device=args.device
    )

    # create MOPO algo
    algo = MOPO(
        sac_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **args.mopo_params
    )

    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        eval_episodes=args.eval_episodes
    )
    
    if not args.load_model:
        # pretrain dynamics model on the whole dataset
        trainer.train_dynamics()
    elif args.load_model:
        # # load pretrained model
        # model_path = "log/walker2d-medium-v2/mopo/seed_1_1120_215452-walker2d_medium_v2_mopo/models/ite_dynamics_model"
        # model_path = "log/walker2d-medium-replay-v2/mopo/seed_1_1204_163535-walker2d_medium_replay_v2_mopo/models/ite_dynamics_model"
        # model_path = "log/walker2d-medium-expert-v2/mopo/seed_1_1204_163511-walker2d_medium_expert_v2_mopo/models/ite_dynamics_model"
        # model_path = "log/walker2d-medium-v2/mopo/seed_1_1208_204229-walker2d_medium_v2_mopo/models/ite_dynamics_model/"
        model_path = "./saved_models/"
        model_file_name = args.task + ".pt"
        trainer.algo.dynamics_model.load_model(model_path, model_file_name)

    # begin train
    trainer.train_policy()


if __name__ == "__main__":
    train()
