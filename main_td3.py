import torch
import numpy as np
from RLlib.td3_agent import DeterministicActor, TwinCritic
from src.environment import DashCamEnv
from src.DADALoader import DADALoader
from RLlib.replay_buffer import ReplayBuffer  # Reuse existing buffer
from trainers.td3_trainer import TD3Trainer

trainer = TD3Trainer(
    actor=actor, 
    critic=critic,
    gamma=cfg.gamma,
    tau=cfg.tau,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2
)

def train_td3(cfg):
    # Initialize environment and dataset
    env = DashCamEnv(cfg)
    dataset = DADALoader(cfg.data_root, cfg.mode, cfg.frame_interval)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TD3 components
    state_dim = 128  # From DRIVE's RAE encoder
    action_dim = 3   # [accident_score, x, y]
    actor = DeterministicActor(state_dim, action_dim).to(device)
    critic = TwinCritic(state_dim, action_dim).to(device)
    replay_buffer = ReplayBuffer(cfg.buffer_size)
    
    # Trainer (from previous TD3Trainer class)
    trainer = TD3Trainer(actor, critic, gamma=cfg.gamma, tau=cfg.tau,
                         policy_noise=0.2, noise_clip=0.5, policy_freq=2)
    
    # Training loop
    for episode in range(cfg.epochs):
        video_data, coord_data, data_info = dataset.sample_batch(cfg.batch_size)
        state = env.set_data(video_data, coord_data, data_info)
        
        for step in range(env.max_steps):
            # Select action with exploration noise
            with torch.no_grad():
                action = actor(state)
                noise = torch.randn_like(action) * cfg.exploration_noise
                action = (action + noise).clamp(-1, 1)
            
            # Step environment
            next_state, reward, _ = env.step(action)
            done = (step == env.max_steps - 1)
            
            # Store transition
            replay_buffer.add(state.cpu(), action.cpu(), reward.cpu(), 
                             next_state.cpu(), done)
            state = next_state
            
            # Train after collecting sufficient samples
            if len(replay_buffer) > cfg.batch_size:
                trainer.update(replay_buffer, cfg.batch_size)

            critic_loss, actor_loss = trainer.update(replay_buffer, cfg.batch_size)
            
            if step % 100 == 0:
                print(f"Step: {step}, Critic Loss: {critic_loss:.3f}, Actor Loss: {actor_loss:.3f}")
        
        # Periodic evaluation
        if episode % cfg.eval_interval == 0:
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
    from cfgs.td3_mlnet import cfg  # Your TD3 config file
    train_td3(cfg)