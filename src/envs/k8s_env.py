import gymnasium as gym
import numpy as np
from gymnasium import spaces
import networkx as nx

class ServicePlacementEnv(gym.Env):
    def __init__(self, num_nodes=20, max_services=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_services = max_services
        
        # Action: Chọn 1 trong N nodes
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # State: [CPU nodes] + [RAM nodes] + [Latency profile (tới cha)] 
        # Tổng length = num_nodes * 3
        self.observation_space = spaces.Box(
            low=-1.0, high=100.0, 
            shape=(self.num_nodes * 3,), 
            dtype=np.float32
        )
        
        self._build_cluster()

    def _build_cluster(self):
        # Tạo cluster 20 nodes, chia làm 2 Zone, mỗi Zone 2 Rack.
        self.node_cpu_cap = np.full(self.num_nodes, 16.0) # 16 vCPU
        self.node_ram_cap = np.full(self.num_nodes, 32.0) # 32 GB RAM
        
        # Ma trận độ trễ: Cùng Rack: 1ms, Khác Rack/Cùng Zone: 5ms, Khác Zone: 20ms
        self.latency_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j: self.latency_matrix[i][j] = 0.1
                elif i//5 == j//5: self.latency_matrix[i][j] = 1.0 # Cùng rack (5 nodes/rack)
                elif i//10 == j//10: self.latency_matrix[i][j] = 5.0 # Cùng zone (10 nodes/zone)
                else: self.latency_matrix[i][j] = 20.0 # Khác zone

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        
        self.current_cpu = self.node_cpu_cap.copy()
        self.current_ram = self.node_ram_cap.copy()
        
        self.num_services_current = np.random.randint(5, self.max_services + 1)
        
        # --- SỬA LỖI Ở ĐÂY ---
        # 1. Tạo đồ thị ngẫu nhiên
        raw_graph = nx.gnp_random_graph(self.num_services_current, 0.4, directed=True)
        
        # 2. Khởi tạo DAG mới và ÉP nó phải chứa đủ số lượng Node (kể cả node cô lập)
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.num_services_current)) 
        # 3. Mới bắt đầu thêm các cạnh (mũi tên 1 chiều) vào
        self.dag.add_edges_from([(u, v) for (u, v) in raw_graph.edges() if u < v])
        
        self.sorted_services = list(nx.topological_sort(self.dag))
        # ---------------------
        
        self.cpu_reqs = np.random.uniform(1.0, 4.0, size=self.num_services_current)
        self.ram_reqs = np.random.uniform(2.0, 8.0, size=self.num_services_current)
        
        self.placements = np.full(self.num_services_current, -1)
        self.current_step = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Khởi tạo profile latency mặc định toàn số 0
        latency_profile = np.zeros(self.num_nodes, dtype=np.float32)
        
        # SỬA LỖI Ở ĐÂY: Chỉ lấy current_svc nếu chưa chạy hết các service
        if self.current_step < len(self.sorted_services):
            current_svc = self.sorted_services[self.current_step]
            parents = list(self.dag.predecessors(current_svc))
            
            if parents: # Nếu service này có cha
                for i in range(self.num_nodes):
                    total_lat = 0
                    for parent in parents:
                        parent_node = self.placements[parent]
                        if parent_node != -1:
                            total_lat += self.latency_matrix[i][parent_node]
                    latency_profile[i] = total_lat

        # Tự động tương thích với cả 2 phiên bản code (có RAM hoặc không có RAM)
        if hasattr(self, 'current_ram'):
            return np.concatenate((self.current_cpu, self.current_ram, latency_profile))
        else:
            return np.concatenate((self.current_node_resources, latency_profile))

    def step(self, action):
        node_idx = action
        current_svc = self.sorted_services[self.current_step]
        cpu_req = self.cpu_reqs[current_svc]
        ram_req = self.ram_reqs[current_svc]
        
        reward = 0
        terminated = False
        
        # 1. Kiểm tra tài nguyên (Resource Constraint)
        if self.current_cpu[node_idx] < cpu_req or self.current_ram[node_idx] < ram_req:
            reward -= 10.0 # Phạt vì tràn RAM/CPU
        else:
            # --- ĐIỂM ĂN TIỀN CHO CV: ANTI-AFFINITY CHECK ---
            # Giả sử: Các service có ID chẵn không được phép nằm cùng Node với nhau
            # (Mô phỏng việc chúng là các Replica của nhau)
            is_anti_affinity_violation = False
            if current_svc % 2 == 0: 
                for placed_svc, placed_node in enumerate(self.placements):
                    if placed_svc != current_svc and placed_svc % 2 == 0 and placed_node == node_idx:
                        is_anti_affinity_violation = True
                        break
            
            if is_anti_affinity_violation:
                reward = -15.0 # Phạt nặng hơn một chút nếu vi phạm High Availability
            else:
                self.current_cpu[node_idx] -= cpu_req
                self.current_ram[node_idx] -= ram_req
                self.placements[current_svc] = node_idx
                
                parents = list(self.dag.predecessors(current_svc))
                lat_cost = sum(self.latency_matrix[node_idx][self.placements[p]] for p in parents if self.placements[p] != -1)
                
                # --- SỬA CÔNG THỨC REWARD Ở ĐÂY ---
                # Thưởng cơ bản +10 điểm cho mỗi lần đặt đúng luật (để tổng điểm vọt lên số Dương)
                # Trừ đi một lượng nhỏ chi phí Latency
                reward = 10.0 - (lat_cost / 5.0)

        self.current_step += 1
        if self.current_step >= len(self.sorted_services):
            terminated = True
            
        return self._get_obs(), reward, terminated, False, {}