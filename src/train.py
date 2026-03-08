import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.k8s_env import ServicePlacementEnv

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Scale hệ thống lên: 50 Máy chủ, 20 Microservices
    env = Monitor(ServicePlacementEnv(num_nodes=50, max_services=20))
    eval_env = Monitor(ServicePlacementEnv(num_nodes=50, max_services=20))

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/cv_project/',
        log_path='./logs/', 
        eval_freq=15000,
        deterministic=True, 
        render=False
    )

    # Tăng trí thông minh (Mạng Neural sâu hơn một chút: 2 lớp 256 nơ-ron)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,          
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/ppo_cv_project/",
        verbose=1,
        device="cuda" # Dùng Card màn hình để chạy
    )

    print("Bắt đầu huấn luyện Đồ án (Dự kiến 1.5 - 2 tiếng)...")
    # Huấn luyện 1,500,000 steps để đảm bảo AI học được luật Anti-Affinity
    model.learn(total_timesteps=1_500_000, callback=eval_callback)
    
    print("Hoàn tất! Model đã lưu tại ./models/cv_project/best_model.zip")