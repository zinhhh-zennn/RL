import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.k8s_env import ServicePlacementEnv

if __name__ == "__main__":
    os.makedirs("models/production", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Scale lên 100 Nodes, 40 Services
    env = Monitor(ServicePlacementEnv(num_nodes=100, max_services=40))
    eval_env = Monitor(ServicePlacementEnv(num_nodes=100, max_services=40))

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/production/',
        log_path='./logs/', 
        eval_freq=20000,
        deterministic=True, 
        render=False
    )

    # --- SỰ KHÁC BIỆT ---
    # Tăng độ sâu mạng Neural lên 3 lớp, mỗi lớp 512 nơ-ron
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU, 
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512])
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0001, # Học chậm hơn để phân tích sâu hơn
        n_steps=4096,         # Nhìn rộng hơn trước khi cập nhật trọng số
        batch_size=256,          
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/ppo_production/",
        verbose=1,
        device="cuda" 
    )

    print("Bắt đầu huấn luyện Production-Grade (Dự kiến 2.5 - 3 tiếng)...")
    model.learn(total_timesteps=3_000_000, callback=eval_callback)
    
    print("Hoàn tất! Model đã lưu tại ./models/production/best_model.zip")