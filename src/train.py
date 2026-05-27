import os
import torch
from sb3_contrib import MaskablePPO as PPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback 
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from envs.k8s_env import ServicePlacementEnv

# --- THÊM HÀM NÀY: Dạy cho Trọng tài cách lấy mặt nạ từ Môi trường ---
def mask_fn(env):
    return env.action_masks()

if __name__ == "__main__":
    os.makedirs("models/production", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # --- SỰ KHÁC BIỆT NẰM Ở ĐÂY ---
    # 1. Bọc mặt nạ cho môi trường Huấn luyện (Train)
    base_env = ServicePlacementEnv(num_nodes=100, max_services=40)
    masked_env = ActionMasker(base_env, mask_fn) # Trọng tài ActionMasker vào việc!
    env = Monitor(masked_env)

    # 2. Bọc mặt nạ cho môi trường Đánh giá (Evaluate)
    base_eval_env = ServicePlacementEnv(num_nodes=100, max_services=40)
    masked_eval_env = ActionMasker(base_eval_env, mask_fn) 
    eval_env = Monitor(masked_eval_env)
    # -------------------------------

    eval_callback = MaskableEvalCallback(
        eval_env, 
        best_model_save_path='./models/production/',
        log_path='./logs/', 
        eval_freq=20000,
        deterministic=True, 
        render=False
    )

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

    print("Bắt đầu huấn luyện Ver 2.0 (Action Masking) - Dự kiến 2.5 - 3 tiếng...")
    model.learn(total_timesteps=3_000_000, callback=eval_callback)
    
    print("Hoàn tất! Model đã lưu tại ./models/production/best_model.zip")