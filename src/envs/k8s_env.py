import gymnasium as gym
import numpy as np
from gymnasium import spaces
import networkx as nx

class ServicePlacementEnv(gym.Env):
    def __init__(self, num_nodes=100, max_services=40):
        super(ServicePlacementEnv, self).__init__()
        self.num_nodes = num_nodes
        self.max_services = max_services
        
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # State bây giờ phức tạp hơn: [CPU_Nodes, RAM_Nodes, Node_Types, Latency_Profile]
        # Kích thước: num_nodes * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=200.0, 
            shape=(self.num_nodes * 4,), 
            dtype=np.float32
        )
        
        self._build_realistic_cluster()

    def _build_realistic_cluster(self):
        """Tạo cụm K8s thực tế với Compute, Memory và General Node Pools"""
        self.node_cpu_cap = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_ram_cap = np.zeros(self.num_nodes, dtype=np.float32)
        self.node_types = np.zeros(self.num_nodes, dtype=np.float32) # 0: General, 1: Compute, 2: Memory
        
        for i in range(self.num_nodes):
            if i % 3 == 0:   # Compute Node (Ví dụ: c5.xlarge)
                self.node_cpu_cap[i], self.node_ram_cap[i], self.node_types[i] = 32.0, 16.0, 1.0
            elif i % 3 == 1: # Memory Node (Ví dụ: r5.xlarge)
                self.node_cpu_cap[i], self.node_ram_cap[i], self.node_types[i] = 16.0, 64.0, 2.0
            else:            # General Node (Ví dụ: m5.xlarge)
                self.node_cpu_cap[i], self.node_ram_cap[i], self.node_types[i] = 16.0, 32.0, 0.0

        # Ma trận Topology: 5 Zones, mỗi Zone 20 Nodes
        self.latency_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j: self.latency_matrix[i][j] = 0.1
                elif i // 20 == j // 20: self.latency_matrix[i][j] = 2.0 # Cùng Zone
                else: self.latency_matrix[i][j] = 15.0 # Khác Zone

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        
        self.current_cpu = self.node_cpu_cap.copy()
        self.current_ram = self.node_ram_cap.copy()
        
        # Sinh số lượng service thực tế cần deploy
        self.num_services_current = np.random.randint(15, self.max_services + 1)
        
        raw_graph = nx.gnp_random_graph(self.num_services_current, 0.3, directed=True)
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.num_services_current)) 
        self.dag.add_edges_from([(u, v) for (u, v) in raw_graph.edges() if u < v])
        
        self.sorted_services = list(nx.topological_sort(self.dag))
        
        # Gán Type cho từng Service (0: Web, 1: API, 2: Database)
        self.service_types = np.random.choice([0, 1, 2], size=self.num_services_current, p=[0.4, 0.4, 0.2])
        self.cpu_reqs = np.zeros(self.num_services_current)
        self.ram_reqs = np.zeros(self.num_services_current)
        self.traffic_weights = np.random.uniform(1.0, 10.0, size=self.num_services_current) # Băng thông (Mbps)
        
        for i in range(self.num_services_current):
            if self.service_types[i] == 0: # Web ăn CPU
                self.cpu_reqs[i], self.ram_reqs[i] = np.random.uniform(4.0, 8.0), np.random.uniform(1.0, 4.0)
            elif self.service_types[i] == 2: # DB ăn RAM
                self.cpu_reqs[i], self.ram_reqs[i] = np.random.uniform(2.0, 4.0), np.random.uniform(8.0, 16.0)
            else: # API trung bình
                self.cpu_reqs[i], self.ram_reqs[i] = np.random.uniform(2.0, 4.0), np.random.uniform(2.0, 4.0)
        
        self.placements = np.full(self.num_services_current, -1)
        self.current_step = 0
        
        # Thêm biến theo dõi Database để làm Masking
        self.node_has_db = np.zeros(self.num_nodes, dtype=bool)

        return self._get_obs(), {}

    def _get_obs(self):
        latency_profile = np.zeros(self.num_nodes, dtype=np.float32)
        if self.current_step < len(self.sorted_services):
            current_svc = self.sorted_services[self.current_step]
            parents = list(self.dag.predecessors(current_svc))
            
            if parents:
                for i in range(self.num_nodes):
                    total_lat = sum(self.latency_matrix[i][self.placements[p]] * self.traffic_weights[p] 
                                    for p in parents if self.placements[p] != -1)
                    latency_profile[i] = total_lat

        return np.concatenate((self.current_cpu, self.current_ram, self.node_types, latency_profile))

    def step(self, action):
        node_idx = action
        current_svc = self.sorted_services[self.current_step]
        svc_type = self.service_types[current_svc]
        cpu_req = self.cpu_reqs[current_svc]
        ram_req = self.ram_reqs[current_svc]
        
        reward = 0
        terminated = False
        
        # 1. Resource Check
        if self.current_cpu[node_idx] < cpu_req or self.current_ram[node_idx] < ram_req:
            reward -= 20.0 # Phạt tràn RAM/CPU
        else:
            # 2. Strict Anti-Affinity cho Database (Type == 2)
            is_anti_affinity_violation = False
            if svc_type == 2: 
                for placed_svc, placed_node in enumerate(self.placements):
                    if placed_svc != current_svc and self.service_types[placed_svc] == 2 and placed_node == node_idx:
                        is_anti_affinity_violation = True
                        break
            
            if is_anti_affinity_violation:
                reward -= 30.0 # Phạt cực nặng nếu 2 DB nằm chung 1 Node
            else:
                self.current_cpu[node_idx] -= cpu_req
                self.current_ram[node_idx] -= ram_req
                self.placements[current_svc] = node_idx

                if svc_type == 2:
                    self.node_has_db[node_idx] = True
                
                # 3. Tính độ trễ nhân với trọng số băng thông
                parents = list(self.dag.predecessors(current_svc))
                lat_cost = sum(self.latency_matrix[node_idx][self.placements[p]] * self.traffic_weights[p] 
                               for p in parents if self.placements[p] != -1)
                
                # 4. Thưởng nếu đặt đúng loại Node (Ví dụ: DB vào Memory Node)
                affinity_bonus = 5.0 if (svc_type == 0 and self.node_types[node_idx] == 1.0) or \
                                        (svc_type == 2 and self.node_types[node_idx] == 2.0) else 0.0
                
                reward = 15.0 + affinity_bonus - (lat_cost / 10.0)

        self.current_step += 1
        if self.current_step >= len(self.sorted_services):
            terminated = True
            
        return self._get_obs(), reward, terminated, False, {}
    
    def action_masks(self):
        """
        Trả về mảng boolean (num_nodes,). True = Được phép đặt, False = Cấm.
        """
        mask = np.ones(self.num_nodes, dtype=bool)
        
        # Nếu đã duyệt hết service thì trả về mask mặc định
        if self.current_step >= len(self.sorted_services):
            return mask

        current_svc = self.sorted_services[self.current_step]
        svc_type = self.service_types[current_svc]
        cpu_req = self.cpu_reqs[current_svc]
        ram_req = self.ram_reqs[current_svc]

        for i in range(self.num_nodes):
            # LUẬT 1: Cấm nếu Node không còn đủ CPU hoặc RAM
            if self.current_cpu[i] < cpu_req or self.current_ram[i] < ram_req:
                mask[i] = False
            
            # LUẬT 2: Cấm nếu là Database (type=2) mà Node đó ĐÃ CHỨA Database rồi
            if svc_type == 2 and self.node_has_db[i]:
                mask[i] = False
        
        # Cơ chế dự phòng: Nếu xui xẻo Cluster bị full 100%, tất cả đều False
        # thì phải mở khóa lại để tránh crash thuật toán (chấp nhận cho AI bị phạt)
        if not mask.any():
            return np.ones(self.num_nodes, dtype=bool)

        return mask