import torch
import copy
import numpy as np

class TD3Trainer:
    def __init__(self, actor, critic, gamma=0.99, tau=0.005, policy_noise=0.2, 
                 noise_clip=0.5, policy_freq=2, actor_lr=1e-4, critic_lr=1e-3):
        """
        Twin Delayed Deep Deterministic Policy Gradients (TD3)
        
        Args:
            actor: Deterministic policy network
            critic: Twin Q-network
            gamma: Discount factor
            tau: Target network update rate
            policy_noise: Std of noise added to target actions
            noise_clip: Noise clipping range
            policy_freq: Frequency of delayed policy updates
        """
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Initialize target networks
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.total_steps = 0

    def update(self, replay_buffer, batch_size):
        """Update policy and value parameters"""
        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Add clipped noise to target actions
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # Compute target Q-values (min of twin critics)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        
        # Critic loss
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + \
                      torch.nn.functional.mse_loss(current_Q2, target_Q)
        
        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = 0.0
        # Delayed policy updates
        if self.total_steps % self.policy_freq == 0:
            # Actor loss (maximize Q1)
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update(self.actor_target, self.actor, tau=self.tau)
            self._soft_update(self.critic_target, self.critic, tau=self.tau)
        self.total_steps += 1
        return critic_loss.item(), actor_loss

    def _soft_update(self, target, source, tau=0.005):
        """Soft update target networks"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        """Save models to file"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        """Load models from file"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])