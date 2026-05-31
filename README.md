## Tối ưu hóa Service Chain Placement trên Kubernetes bằng Reinforcement Learning

## Giới thiệu

Đây là đồ án môn **Học máy tăng cường cho hệ thống mạng**, tập trung vào bài toán **tối ưu hóa vị trí đặt các microservices trong chuỗi dịch vụ trên Kubernetes**.

Trong kiến trúc microservices, một ứng dụng thường gồm nhiều service giao tiếp với nhau. Nếu các service phụ thuộc nhau bị đặt quá xa nhau trên cụm Kubernetes, độ trễ mạng có thể tăng. Ngược lại, nếu gom quá nhiều service vào cùng một node hoặc cùng một vùng, hệ thống có thể bị mất cân bằng tải và tạo ra rủi ro Single Point of Failure.

Đề tài xây dựng một mô hình mô phỏng **AI Scheduler** sử dụng thuật toán **Proximal Policy Optimization (PPO)** trong Reinforcement Learning. Mô hình học cách chọn node phù hợp để triển khai service, đồng thời kết hợp **Invalid Action Masking** để loại bỏ các hành động không hợp lệ như node thiếu CPU/RAM hoặc vi phạm ràng buộc Anti-Affinity.

## Mục tiêu

- Mô hình hóa bài toán Service Placement dưới dạng bài toán Reinforcement Learning.
- Xây dựng môi trường mô phỏng cụm Kubernetes.
- Huấn luyện PPO Agent để chọn node triển khai service.
- Đảm bảo các ràng buộc cơ bản như CPU, RAM và Anti-Affinity.
- So sánh kết quả với các chiến lược baseline như Random, First-Fit và Greedy Latency.

## Thành phần chính

- **Kubernetes Simulation Environment**: Mô phỏng trạng thái cụm Kubernetes, tài nguyên node và độ trễ mạng.
- **PPO Agent**: Tác tử học máy tăng cường dùng để đưa ra quyết định chọn node.
- **Action Masking**: Lọc các node không hợp lệ trước khi Agent chọn hành động.
- **Telemetry & Evaluation**: Ghi log, phân tích reward và trực quan hóa kết quả bằng heatmap/TensorBoard.

## Ý nghĩa đề tài

Đề tài cho thấy Reinforcement Learning có thể được áp dụng để hỗ trợ bài toán lập lịch trong môi trường cloud-native. Thay vì chỉ tối ưu một mục tiêu như độ trễ, mô hình có thể học cách cân bằng giữa độ trễ, tài nguyên và tính sẵn sàng cao của hệ thống.

## Lưu ý

Dự án hiện tại là mô hình proof-of-concept trong môi trường mô phỏng, chưa phải custom scheduler chạy trực tiếp trên Kubernetes production.
