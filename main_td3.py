import torch
import numpy as np
from RLlib.td3_agent import DeterministicActor, TwinCritic
from src.environment import DashCamEnv
from src.DADALoader import DADALoader
from RLlib.replay_buffer import ReplayMemory, ReplayMemoryGPU  # Reuse existing buffer
from trainers.td3_trainer import TD3Trainer
import yaml
from src.DADALoader import setup_dataloader
from src.saliency.mlnet import MLNet  # Import MLNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("cfgs/td3_mlnet.yml", "r") as f:
    cfg = yaml.safe_load(f)

def train_td3(cfg):
    # Initialize environment and dataset
    if cfg["saliency"] == 'MLNet':
        observe_model = MLNet(cfg["input_shape"]).to(device)
    # Check if pretrained weights are specified
        if cfg["env_model"]:
            assert os.path.exists(cfg["env_model"]), "Checkpoint directory does not exist! %s"%(cfg["env_model"])
            ckpt = torch.load(cfg["env_model"])
            observe_model.load_state_dict(ckpt['model'])
        
    env = DashCamEnv(cfg, device, observe_model)
    #dataset = DADALoader(cfg.data_root, cfg.mode, cfg.frame_interval)
    dataset = DADALoader(cfg["data_path"], cfg["mode"], cfg["frame_interval"])

    # TD3 components
    state_dim = 128  # From DRIVE's RAE encoder
    action_dim = 3   # [accident_score, x, y]
    actor = DeterministicActor(state_dim, action_dim).to(device)
    critic = TwinCritic(state_dim, action_dim).to(device)
    #replay_buffer = ReplayMemory(cfg.buffer_size)
    replay_buffer = ReplayMemoryGPU(cfg, device=device)
    #replay_buffer = ReplayMemoryGPU(replay_size=cfg["replay_size"], device=device)

    trainer = TD3Trainer(
    actor=actor.to(device), 
    critic=critic.to(device),
    gamma=cfg["gamma"],
    tau=cfg["tau"],
    policy_noise=cfg["policy_noise"],
    noise_clip=cfg["noise_clip"],
    policy_freq=cfg["policy_freq"]
    )
    
    # Training loop
    for episode in range(cfg['epochs']):
        #video_data, coord_data, data_info = dataset.sample_batch(cfg['batch_size'])
        dataloader = setup_dataloader(cfg)  # ✅ Get the data loader
        for batch in dataloader:
            video_data, coord_data, data_info, *_ = batch
            break  # ✅ Only take one batch per episode
        state = env.set_data(video_data, coord_data, data_info)
        
        for step in range(env.max_steps):
            # Select action with exploration noise
            with torch.no_grad():
                action = actor(state)
                #noise = torch.randn_like(action) * cfg.exploration_noise
                noise = torch.randn_like(action) * cfg["exploration_noise"]
                action = (action + noise).clamp(-1, 1)
            
            # Step environment
            next_state, reward, _ = env.step(action)
            done = (step == env.max_steps - 1)
            
            # Store transitiong
            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )
            state = next_state
            
            # Train after collecting sufficient samples
            if len(replay_buffer) > cfg['batch_size']:
                critic_loss, actor_loss = trainer.update(replay_buffer, cfg['batch_size'])
                
            
            if step % 100 == 0:
                print(f"Step: {step}, Critic Loss: {critic_loss:.3f}, Actor Loss: {actor_loss:.3f}")
        
        # Periodic evaluation
        if episode % cfg["eval_interval"] == 0:
            test_performance(actor, env, device)

def test_performance(actor, env, device):
    # Run agent without exploration noise
    with torch.no_grad():
        state = env.reset()
        total_reward = 0
        for _ in range(env.max_steps):
            action = actor(state)
            state, reward, _ = env.step(action, isTraining=False)
            total_reward += reward.mean().item()
    print(f"Test Reward: {total_reward:.2f}")

if __name__ == "__main__":
    #from cfgs.td3_mlnet import cfg  # Your TD3 config file
    train_td3(cfg)