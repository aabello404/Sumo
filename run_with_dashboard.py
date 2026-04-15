"""
WORKFLOW - HOW TO RUN:
Step 1: Ensure models are trained (or it runs random actions)
Step 2: python run_with_dashboard.py
Step 3: open http://localhost:8000
"""
import os
import sys
import time
import json
import threading
import argparse
import random  # Added this import for the random policy
import uvicorn
from stable_baselines3 import PPO
from traffic_env import MultiAgentSUMOEnv, SUMO_CFG, MAX_STEPS

def load_models(tl_ids):
    models = {}
    for i, tl in enumerate(tl_ids):
        best_path = f"models/agent_{i}/best/best_model.zip"
        final_path = f"models/agent_{i}/final_model.zip"
        
        if os.path.exists(best_path):
            models[tl] = PPO.load(best_path)
            print(f"Loaded BEST model for {tl} (agent_{i})")
        elif os.path.exists(final_path):
            models[tl] = PPO.load(final_path)
            print(f"Loaded FINAL model for {tl} (agent_{i})")
        else:
            models[tl] = None
            print(f"No model found for {tl} (agent_{i}), using RANDOM policy")
    return models

def run_simulation(push_fn, args):
    env = MultiAgentSUMOEnv(SUMO_CFG, args.max_steps, args.gui)
    
    episode = 0
    while True:
        episode += 1
        obs_dict = env.reset()
        tl_ids = env.tl_ids
        models = load_models(tl_ids)
        
        ep_rewards = {tl: 0.0 for tl in tl_ids}
        
        for step in range(args.max_steps):
            actions = {}
            for tl in tl_ids:
                if models[tl] is not None:
                    action, _ = models[tl].predict(obs_dict[tl], deterministic=True)
                    actions[tl] = int(action)
                else:
                    # FIXED: Using Python's built-in random instead of broken SUMO command
                    actions[tl] = random.randint(0, 1)
                    
            obs_dict, reward_dict, done_dict, _ = env.step(actions)
            
            for tl, r in reward_dict.items():
                ep_rewards[tl] += r
                
            state_payload = env.get_dashboard_state()
            state_payload["actions"] = actions
            state_payload["rewards"] = reward_dict
            state_payload["ep_rewards"] = ep_rewards
            state_payload["episode"] = episode
            
            try:
                if os.path.exists("training_stats.json"):
                    with open("training_stats.json", "r") as f:
                        state_payload["training_stats"] = json.load(f)
            except:
                state_payload["training_stats"] = {}
                
            push_fn(state_payload)
            time.sleep(args.step_delay)
            
            if all(done_dict.values()):
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--step-delay", type=float, default=0.05)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()

    from dashboard_server import push_state, app

    sim_thread = threading.Thread(target=run_simulation, args=(push_state, args), daemon=True)
    sim_thread.start()

    if not args.no_server:
        print("Starting Dashboard server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")