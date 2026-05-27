import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sb3_contrib import MaskablePPO as PPO
from sb3_contrib.common.wrappers import ActionMasker
from envs.k8s_env import ServicePlacementEnv
import os

os.makedirs("results", exist_ok=True)

# Dạy cho hệ thống cách lấy mặt nạ từ môi trường
def mask_fn(env):
    return env.action_masks()

def run_random(env):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def run_first_fit(env):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    num_nodes = env.unwrapped.num_nodes
    while not terminated:
        cpu_avail = obs[0 : num_nodes]
        ram_avail = obs[num_nodes : 2 * num_nodes]
        current_svc = env.unwrapped.sorted_services[env.unwrapped.current_step]
        cpu_req = env.unwrapped.cpu_reqs[current_svc]
        ram_req = env.unwrapped.ram_reqs[current_svc]
        
        action = 0
        for i in range(num_nodes):
            if cpu_avail[i] >= cpu_req and ram_avail[i] >= ram_req:
                action = i
                break
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def run_greedy_latency(env):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    num_nodes = env.unwrapped.num_nodes
    while not terminated:
        cpu_avail = obs[0 : num_nodes]
        ram_avail = obs[num_nodes : 2 * num_nodes]
        latency_profile = obs[3 * num_nodes : 4 * num_nodes]
        
        current_svc = env.unwrapped.sorted_services[env.unwrapped.current_step]
        cpu_req = env.unwrapped.cpu_reqs[current_svc]
        ram_req = env.unwrapped.ram_reqs[current_svc]
        
        best_node, min_lat = -1, float('inf')
        for i in range(num_nodes):
            if cpu_avail[i] >= cpu_req and ram_avail[i] >= ram_req:
                if latency_profile[i] < min_lat:
                    min_lat = latency_profile[i]
                    best_node = i
        action = best_node if best_node != -1 else 0
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def run_rl_agent(env, model):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        # Lấy mặt nạ từ môi trường (thông qua unwrapped để xuyên qua ActionMasker)
        action_masks = env.unwrapped.action_masks()
        # Ép AI phải nhìn vào mặt nạ khi ra quyết định
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def plot_cluster_utilization(env, title, filename):
    """Vẽ Heatmap thể hiện tải của 100 Nodes"""
    cpu_used = env.unwrapped.node_cpu_cap - env.unwrapped.current_cpu
    cpu_percent = (cpu_used / env.unwrapped.node_cpu_cap) * 100
    
    # Reshape thành lưới 10x10 cho đẹp
    cpu_matrix = cpu_percent.reshape(10, 10)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cpu_matrix, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={'label': '% CPU Utilization'})
    plt.title(title)
    plt.xlabel("Rack ID")
    plt.ylabel("Zone ID")
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Bọc môi trường bằng Trọng tài ActionMasker
    base_env = ServicePlacementEnv(num_nodes=100, max_services=40) 
    env = ActionMasker(base_env, mask_fn)
    
    model_path = "models/production/best_model.zip"
    model = PPO.load(model_path)
    
    # 1. Vẽ Boxplot So sánh
    n_episodes = 100
    results = {"Random": [], "First-Fit (K8s)": [], "Greedy Latency": [], "RL Agent (PPO)": []}
    
    print("Đang thi đấu 100 ván...")
    for i in range(n_episodes):
        env.reset(seed=i); results["Random"].append(run_random(env))
        env.reset(seed=i); results["First-Fit (K8s)"].append(run_first_fit(env))
        env.reset(seed=i); results["Greedy Latency"].append(run_greedy_latency(env))
        env.reset(seed=i); results["RL Agent (PPO)"].append(run_rl_agent(env, model))

    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    df.boxplot()
    plt.title("So Sánh Điểm Thưởng (Scale: 100 Nodes, 40 Services)")
    plt.ylabel("Reward")
    plt.savefig("results/boxplot_production.png", dpi=300)
    
    # 2. Sinh Heatmap trực quan cho 1 ván cụ thể (Ván số 42)
    env.reset(seed=42)
    run_greedy_latency(env)
    plot_cluster_utilization(env, "Node CPU Utilization - Greedy (Overloaded spots)", "heatmap_greedy.png")
    
    env.reset(seed=42)
    run_rl_agent(env, model)
    plot_cluster_utilization(env, "Node CPU Utilization - RL Agent (Balanced Load)", "heatmap_rl.png")
    
    print("Hoàn tất! Hãy mở thư mục results/ để xem ảnh Heatmap và Boxplot.")