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

# CẬP NHẬT: Các hàm run_* giờ trả về cả mảng điểm tích lũy từng bước (để vẽ Line Chart)
def run_random(env):
    obs, _ = env.reset()
    total_reward = 0
    step_rewards = []
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        step_rewards.append(total_reward)
    return total_reward, step_rewards

def run_first_fit(env):
    obs, _ = env.reset()
    total_reward = 0
    step_rewards = []
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
        step_rewards.append(total_reward)
    return total_reward, step_rewards

def run_greedy_latency(env):
    obs, _ = env.reset()
    total_reward = 0
    step_rewards = []
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
        step_rewards.append(total_reward)
    return total_reward, step_rewards

def run_rl_agent(env, model):
    obs, _ = env.reset()
    total_reward = 0
    step_rewards = []
    terminated = False
    while not terminated:
        action_masks = env.unwrapped.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        step_rewards.append(total_reward)
    return total_reward, step_rewards

def plot_cluster_utilization(env, title, filename, resource="CPU"):
    """Vẽ Heatmap thể hiện tải của 100 Nodes (hỗ trợ cả CPU và RAM)"""
    if resource == "CPU":
        used = env.unwrapped.node_cpu_cap - env.unwrapped.current_cpu
        percent = (used / env.unwrapped.node_cpu_cap) * 100
        cmap_color = "YlOrRd"
    else:
        used = env.unwrapped.node_ram_cap - env.unwrapped.current_ram
        percent = (used / env.unwrapped.node_ram_cap) * 100
        cmap_color = "PuBu" # RAM dùng màu xanh cho dễ phân biệt
        
    matrix = percent.reshape(10, 10)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap=cmap_color, cbar_kws={'label': f'% {resource} Utilization'})
    plt.title(title)
    plt.xlabel("Rack ID")
    plt.ylabel("Zone ID")
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

def get_std_dev(env):
    """Tính độ lệch chuẩn tải trọng CPU (Dùng để đo lường Cân bằng tải)"""
    cpu_used = env.unwrapped.node_cpu_cap - env.unwrapped.current_cpu
    cpu_percent = (cpu_used / env.unwrapped.node_cpu_cap) * 100
    return np.std(cpu_percent)

if __name__ == "__main__":
    base_env = ServicePlacementEnv(num_nodes=100, max_services=40) 
    env = ActionMasker(base_env, mask_fn)
    model = PPO.load("models/production/best_model.zip")
    
    n_episodes = 100
    results = {"Random": [], "First-Fit (K8s)": [], "Greedy Latency": [], "RL Agent (PPO)": []}
    
    # Lưu lại hành trình của ván số 42 để vẽ biểu đồ chi tiết
    trajectories = {}
    
    print("Đang thi đấu 100 ván...")
    for i in range(n_episodes):
        env.reset(seed=i); rew_rand, traj_rand = run_random(env); results["Random"].append(rew_rand)
        env.reset(seed=i); rew_ff, traj_ff = run_first_fit(env); results["First-Fit (K8s)"].append(rew_ff)
        env.reset(seed=i); rew_gr, traj_gr = run_greedy_latency(env); results["Greedy Latency"].append(rew_gr)
        env.reset(seed=i); rew_rl, traj_rl = run_rl_agent(env, model); results["RL Agent (PPO)"].append(rew_rl)
        
        if i == 42:
            trajectories = {"Random": traj_rand, "First-Fit (K8s)": traj_ff, 
                            "Greedy Latency": traj_gr, "RL Agent (PPO)": traj_rl}

    # BIỂU ĐỒ 1: Boxplot Tổng quát (Như cũ)
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    df.boxplot()
    plt.title("So Sánh Điểm Thưởng (Scale: 100 Nodes, 40 Services)")
    plt.ylabel("Reward")
    plt.savefig("results/boxplot_production.png", dpi=300)
    plt.close()
    
    # BIỂU ĐỒ 2: Biểu đồ đường (Line Chart) - Cumulative Reward
    plt.figure(figsize=(10, 6))
    for name, traj in trajectories.items():
        plt.plot(traj, label=name, linewidth=2)
    plt.title("Tiến trình Tích lũy Điểm thưởng (Cumulative Reward Trajectory - Ván 42)")
    plt.xlabel("Bước quyết định (Từng Service được đặt)")
    plt.ylabel("Tổng điểm thưởng tích lũy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("results/linechart_cumulative_reward.png", dpi=300)
    plt.close()

    # Thu thập dữ liệu Độ lệch chuẩn để vẽ Biểu đồ Cân bằng tải
    std_devs = {}
    env.reset(seed=42)
    run_first_fit(env); std_devs["First-Fit"] = get_std_dev(env)
    env.reset(seed=42)
    run_greedy_latency(env); std_devs["Greedy Latency"] = get_std_dev(env)
    env.reset(seed=42)
    run_rl_agent(env, model); std_devs["RL Agent (PPO)"] = get_std_dev(env)
    
    # BIỂU ĐỒ 3: Biểu đồ Cột (Bar Chart) - Load Balancing Variance
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(std_devs.keys()), y=list(std_devs.values()), hue=list(std_devs.keys()), palette="viridis", legend=False)
    plt.title("Độ lệch chuẩn tải trọng CPU (Load Balancing Variance)\n* Càng thấp chứng tỏ hệ thống phân bổ càng đồng đều *")
    plt.ylabel("Độ lệch chuẩn % CPU (Standard Deviation)")
    plt.savefig("results/barchart_load_balancing.png", dpi=300)
    plt.close()

    # BIỂU ĐỒ 4 & 5: Heatmap CPU & RAM cho Agent RL
    env.reset(seed=42)
    run_rl_agent(env, model)
    plot_cluster_utilization(env, "Node CPU Utilization - RL Agent", "heatmap_rl_cpu.png", resource="CPU")
    plot_cluster_utilization(env, "Node RAM Utilization - RL Agent", "heatmap_rl_ram.png", resource="RAM")
    
    # BIỂU ĐỒ 6: Heatmap CPU cho Greedy (Để so sánh)
    env.reset(seed=42)
    run_greedy_latency(env)
    plot_cluster_utilization(env, "Node CPU Utilization - Greedy (Overloaded spots)", "heatmap_greedy_cpu.png", resource="CPU")

    print("Hoàn tất! Hãy mở thư mục results/ để xem trọn bộ 6 biểu đồ phân tích chuyên sâu.")