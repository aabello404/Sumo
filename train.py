"""
WORKFLOW - HOW TO RUN:
Step 1: Train agents (auto-detects how many TLs exist)
Step 2: python train.py --timesteps 300000
"""
import os
import json
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from traffic_env import SingleAgentWrapper, detect_num_tls, SUMO_CFG, MAX_STEPS

class RewardLoggerCallback(BaseCallback):
    def __init__(self, agent_id, stats_file="training_stats.json", verbose=0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.stats_file = stats_file
        self.episode_rewards = []
        self.current_ep_reward = 0

    def _on_step(self) -> bool:
        self.current_ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(float(self.current_ep_reward))
            self.current_ep_reward = 0
            self._save_stats()
        return True

    def _save_stats(self):
        stats = {}
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r") as f:
                    stats = json.load(f)
            except:
                pass
                
        agent_key = f"agent_{self.agent_id}"
        stats[agent_key] = {
            "rewards": self.episode_rewards[-100:], 
            "mean_reward": sum(self.episode_rewards[-10:]) / max(1, len(self.episode_rewards[-10:])),
            "total_steps": self.num_timesteps
        }
        
        with open(self.stats_file, "w") as f:
            json.dump(stats, f)

# Added current_num and total_agents to track overall macro progress
def train_agent(agent_idx, total_timesteps, current_num=1, total_agents=1):
    print(f"\n========== TRAINING AGENT {agent_idx} (Agent {current_num} of {total_agents}) ==========")
    env = DummyVecEnv([lambda: SingleAgentWrapper(agent_idx, SUMO_CFG, MAX_STEPS, False)])
    eval_env = DummyVecEnv([lambda: SingleAgentWrapper(agent_idx, SUMO_CFG, MAX_STEPS, False)])

    save_dir = f"models/agent_{agent_idx}"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_model"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best",
        log_path=save_dir,
        eval_freq=25000,
        deterministic=True,
        render=False
    )
    
    reward_callback = RewardLoggerCallback(agent_idx)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=f"./logs/agent_{agent_idx}/"
    )

    # progress_bar=True activates the visual tracker!
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, reward_callback],
        progress_bar=True 
    )
    
    model.save(f"{save_dir}/final_model")
    env.close()
    eval_env.close()

def evaluate_agents():
    num_tls = detect_num_tls()
    print(f"Evaluating {num_tls} agents...")
    for i in range(num_tls):
        env = SingleAgentWrapper(i, SUMO_CFG, MAX_STEPS, False)
        model_path = f"models/agent_{i}/best/best_model.zip"
        if not os.path.exists(model_path):
            print(f"No model found for agent {i} at {model_path}")
            continue
            
        model = PPO.load(model_path)
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done: break
        print(f"Agent {i} Total Evaluation Reward: {total_reward}")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=int, default=-1, help="Agent index to train (-1 for all)")
    parser.add_argument("--timesteps", type=int, default=300000, help="Total timesteps per agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    args = parser.parse_args()

    if args.eval:
        evaluate_agents()
    else:
        num_tls = detect_num_tls()
        print(f"Detected {num_tls} traffic lights.")
        
        if os.path.exists("training_stats.json"):
            os.remove("training_stats.json")
            
        if args.agent == -1:
            # Force the script to only train a maximum of 4 agents
            limit = min(4, num_tls)
            print(f"Limiting training to {limit} agents so your CPU doesn't melt...")
            for i in range(limit):
                # We pass 'i+1' as the current agent number, and 'limit' as the total
                train_agent(i, args.timesteps, current_num=i+1, total_agents=limit)
        else:
            if 0 <= args.agent < num_tls:
                train_agent(args.agent, args.timesteps, current_num=1, total_agents=1)
            else:
                print(f"Invalid agent index. Must be between 0 and {num_tls-1}")