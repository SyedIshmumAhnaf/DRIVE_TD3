import torch
import numpy as np
from RLlib.td3_agent import DeterministicActor, TwinCritic
from src.environment import DashCamEnv
from src.DADALoader import DADALoader
from RLlib.replay_buffer import ReplayMemory, ReplayMemoryGPU  # Reuse existing buffer
from trainers.td3_trainer import TD3Trainer
import yaml
from src.DADALoader import setup_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("cfgs/td3_mlnet.yml", "r") as f:
    cfg = yaml.safe_load(f)

def train_td3(cfg):
    # Initialize environment and dataset
    #env = DashCamEnv(cfg)
    env = DashCamEnv(cfg, device)
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
    
    # Trainer (from previous TD3Trainer class)
    #trainer = TD3Trainer(actor, critic, gamma=cfg['gamma'], tau=cfg['tau'],
    #                     policy_noise=0.2, noise_clip=0.5, policy_freq=2)
    
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
            replay_buffer.add(
                state.to(device),
                action.to(device),
                reward.to(device),
                next_state.to(device),
                done
            )
            state = next_state
            
            # Train after collecting sufficient samples
            if len(replay_buffer) > cfg['batch_size']:
                trainer.update(replay_buffer, cfg['batch_size'])

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