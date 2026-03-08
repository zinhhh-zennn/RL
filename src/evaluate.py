# src/evaluate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.k8s_env import ServicePlacementEnv
import os

# Tạo thư mục lưu biểu đồ
os.makedirs("results", exist_ok=True)

# ==========================================
# CÁC THUẬT TOÁN ĐỐI THỦ (BASELINES)
# ==========================================

def run_random(env):
    """Xếp bừa (Random): Dùng để làm mốc tệ nhất."""
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def run_first_fit(env):
    """
    First-Fit (Giống K8s Default nhất): 
    Duyệt từ Node 0 đến N, thấy Node nào còn đủ CPU/RAM thì nhét vào ngay.
    Hoàn toàn mù tịt về Network Latency.
    """
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    num_nodes = env.unwrapped.num_nodes
    
    while not terminated:
        # Tách State ra để đọc tài nguyên
        cpu_avail = obs[0 : num_nodes]
        ram_avail = obs[num_nodes : 2 * num_nodes]
        
        current_svc = env.unwrapped.sorted_services[env.unwrapped.current_step]
        cpu_req = env.unwrapped.cpu_reqs[current_svc]
        ram_req = env.unwrapped.ram_reqs[current_svc]
        
        action = 0 # Mặc định nếu full hết
        for i in range(num_nodes):
            if cpu_avail[i] >= cpu_req and ram_avail[i] >= ram_req:
                action = i
                break # Tìm thấy là chọn luôn
                
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def run_greedy_latency(env):
    """
    Greedy (Tham lam): 
    Chọn Node có độ trễ (latency) tới các service cha là THẤP NHẤT.
    Thuật toán này rất mạnh, nhưng hay làm nghẽn 1 node.
    """
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    num_nodes = env.unwrapped.num_nodes
    
    while not terminated:
        cpu_avail = obs[0 : num_nodes]
        ram_avail = obs[num_nodes : 2 * num_nodes]
        latency_profile = obs[2 * num_nodes : 3 * num_nodes]
        
        current_svc = env.unwrapped.sorted_services[env.unwrapped.current_step]
        cpu_req = env.unwrapped.cpu_reqs[current_svc]
        ram_req = env.unwrapped.ram_reqs[current_svc]
        
        best_node = -1
        min_lat = float('inf')
        
        for i in range(num_nodes):
            # Chỉ xét các node đủ tài nguyên
            if cpu_avail[i] >= cpu_req and ram_avail[i] >= ram_req:
                if latency_profile[i] < min_lat:
                    min_lat = latency_profile[i]
                    best_node = i
                    
        action = best_node if best_node != -1 else 0
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def run_rl_agent(env, model):
    """AI của bạn (PPO)"""
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
    return total_reward

# ==========================================
# HÀM MAIN: CHẠY THỰC NGHIỆM VÀ VẼ BIỂU ĐỒ
# ==========================================
if __name__ == "__main__":
    print("Đang tải môi trường và mô hình...")
    # Dùng đúng tham số bạn đã train
    env = ServicePlacementEnv(num_nodes=50, max_services=20) 
    
    # Load model tốt nhất (sửa đường dẫn nếu cần)
    model_path = "models/cv_project/best_model.zip"
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy model tại {model_path}. Vui lòng chạy train.py trước!")
        exit()
        
    model = PPO.load(model_path)
    
    n_episodes = 200 # Chạy thử 200 đồ thị mạng khác nhau
    
    results = {
        "Random": [],
        "First-Fit (K8s)": [],
        "Greedy Latency": [],
        "RL Agent (PPO)": []
    }
    
    print(f"Bắt đầu đánh giá trên {n_episodes} kịch bản (Episodes)...")
    for i in range(n_episodes):
        # Đặt chung một seed cho mỗi episode để 4 thuật toán giải ĐÚNG 1 bài toán giống nhau
        env.reset(seed=i)
        results["Random"].append(run_random(env))
        
        env.reset(seed=i)
        results["First-Fit (K8s)"].append(run_first_fit(env))
        
        env.reset(seed=i)
        results["Greedy Latency"].append(run_greedy_latency(env))
        
        env.reset(seed=i)
        results["RL Agent (PPO)"].append(run_rl_agent(env, model))
        
        if (i+1) % 50 == 0:
            print(f"Đã hoàn thành {i+1}/{n_episodes} kịch bản...")

    # Xuất ra file Excel/CSV để sau này copy vào Word làm báo cáo
    df = pd.DataFrame(results)
    df.to_csv("results/evaluation_scores.csv", index=False)
    print("\nĐã lưu điểm số chi tiết vào: results/evaluation_scores.csv")
    
    # In điểm trung bình ra màn hình
    print("\n=== ĐIỂM SỐ TRUNG BÌNH (Càng cao càng tốt) ===")
    print(df.mean().round(2))

    # --- VẼ BIỂU ĐỒ BOXPLOT ---
    # Boxplot là biểu đồ học thuật chuẩn nhất để đo sự ổn định (Stability)
    plt.figure(figsize=(10, 6))
    df.boxplot(column=["Random", "First-Fit (K8s)", "Greedy Latency", "RL Agent (PPO)"])
    plt.title("So sánh Hiệu suất Đặt Microservices (Service Placement)")
    plt.ylabel("Tổng Điểm Thưởng (Reward)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = "results/boxplot_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Đã lưu biểu đồ tại: {plot_path}")
    plt.show()