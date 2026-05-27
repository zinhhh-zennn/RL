import numpy as np
from sb3_contrib import MaskablePPO as PPO
from envs.k8s_env import ServicePlacementEnv

def run_demo():
    print("\n" + "="*90)
    print("🚀 DEMO ĐỒ ÁN: TỐI ƯU HÓA MICROSERVICES BẰNG RL (AI SCHEDULER)")
    print("="*90)

    # 1. Khởi tạo môi trường 100 Nodes và Load bộ não AI đã train
    env = ServicePlacementEnv(num_nodes=100, max_services=40)
    model = PPO.load("models/production/best_model.zip")

    # Đặt seed=42 cố định để lúc quay video kịch bản không bị nhảy lộn xộn
    obs, _ = env.reset(seed=42)
    terminated = False

    print("\n[KỊCH BẢN]: KIỂM TRA TÍNH THÔNG MINH TRONG PHÂN BỔ TÀI NGUYÊN VÀ ANTI-AFFINITY")
    print(f"{'Service ID':<12} | {'Loại Service':<18} | {'Đặt tại Node':<14} | {'Loại Node':<12} | {'Nhận xét'}")
    print("-" * 90)

    db_nodes = [] # Lưu vết các Node đã đặt Database để kiểm tra luật High Availability

    total_reward = 0

    while not terminated:
        # Lấy thông tin service hiện tại từ môi trường
        current_step = env.unwrapped.current_step
        current_svc = env.unwrapped.sorted_services[current_step]
        svc_type = env.unwrapped.service_types[current_svc]

        # Đổi mã số thành tên cho dễ đọc trên log
        svc_name = "Web (Cần CPU)" if svc_type == 0 else "API (General)" if svc_type == 1 else "Database (Cần RAM)"

        # Lấy mặt nạ từ môi trường
        action_masks = env.unwrapped.action_masks()

        # AI ra quyết định dựa trên mặt nạ (không bao giờ chọn vào Node bị cấm)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        node_idx = action
        node_type = env.unwrapped.node_types[node_idx]

        node_name = "General" if node_type == 0.0 else "Compute" if node_type == 1.0 else "Memory"

        # Tự động đánh giá quyết định của AI để in ra màn hình
        status = "✅ Hợp lệ"
            
        # Đánh giá việc phân bổ đúng loại Node (Tối ưu phần cứng)
        if svc_type == 0 and node_type == 1.0:
            status = "⭐ Tối ưu (Web -> Compute)"
        elif svc_type == 2 and node_type == 2.0:
            status = "⭐ Tối ưu (DB -> Memory)"
                
        # Đánh giá luật Anti-Affinity (Không đặt chung Node cho Database)
        if svc_type == 2:
            if node_idx in db_nodes:
                status = "❌ LỖI ANTI-AFFINITY" # AI đã train chuẩn sẽ không bao giờ vướng lỗi này
            else:
                db_nodes.append(node_idx)
                status += " (Đảm bảo HA)"

        # In log ra màn hình
        print(f"Service {current_svc:<4} | {svc_name:<18} | Node {node_idx:<10} | {node_name:<12} | {status}")
            
        # Thực thi hành động và bước sang trạng thái tiếp theo
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward

    print("-" * 90)
    print(f"Tổng số điểm Reward đạt được trong kịch bản này: {total_reward:.2f}")
    print("="*90 + "\n")

if __name__ == "__main__":
    run_demo()