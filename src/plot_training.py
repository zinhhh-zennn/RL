import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_training_curves(log_dir="./logs/ppo_production/", output_file="results/training_curves.png"):
    print("Đang đọc dữ liệu từ TensorBoard logs...")
    
    # Tìm file log mới nhất trong thư mục
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    latest_log_dir = max(subdirs, key=os.path.getmtime) if subdirs else log_dir
    
    # Khởi tạo bộ đọc
    event_acc = EventAccumulator(latest_log_dir)
    event_acc.Reload()

    # Trích xuất dữ liệu (Thay đổi tags tùy theo log của SB3)
    tags = event_acc.Tags()['scalars']
    
    # Các metrics quan trọng cần vẽ cho PPO
    metrics = {
        'Reward': 'rollout/ep_rew_mean' if 'rollout/ep_rew_mean' in tags else None,
        'Policy Loss': 'train/policy_gradient_loss' if 'train/policy_gradient_loss' in tags else None,
        'Value Loss': 'train/value_loss' if 'train/value_loss' in tags else None,
        'Approx KL Divergence': 'train/approx_kl' if 'train/approx_kl' in tags else None
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    
    titles = [
        "(a) Training Reward (Điểm thưởng)", 
        "(b) Policy Loss (Mất mát Chính sách)", 
        "(c) Value Loss (Mất mát Giá trị)", 
        "(d) KL Divergence (Độ phân kỳ)"
    ]

    for i, (metric_name, tag) in enumerate(metrics.items()):
        ax = axes[i]
        if tag:
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            # Vẽ đường nét mảnh cho dữ liệu gốc
            ax.plot(steps, values, alpha=0.3, color='#1f77b4')
            
            # Làm mượt (Smoothing) bằng đường nét đậm
            smoothed = []
            alpha = 0.9 # Hệ số làm mượt
            for val in values:
                if not smoothed:
                    smoothed.append(val)
                else:
                    smoothed.append(smoothed[-1] * alpha + val * (1 - alpha))
            
            ax.plot(steps, smoothed, color='#005293', linewidth=1.5)
            
        ax.set_title(titles[i], fontsize=14, pad=15)
        ax.set_xlabel("Timesteps")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Xóa viền trên và viền phải cho giống phong cách học thuật
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Hoàn tất! Đã lưu biểu đồ tại: {output_file}")

if __name__ == "__main__":
    plot_training_curves()