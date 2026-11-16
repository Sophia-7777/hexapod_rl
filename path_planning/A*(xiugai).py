import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# =============================
# 基本参数
# =============================
start = (0, 0)
goal = (10, 10)
grid_size = (20, 20)

obstacle = [
    (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
    (3, 0), (3, 1), (3, 3), (3, 4), (3, 5)
]

# =============================
# A* 全局变量
# =============================
open_set = [(start[0], start[1], 0, 0, 0)]
closed_set = []
visit_order = []

g_cost = {start: 0}
came_from = {}

# =============================
# 启发函数
# =============================
def Euclidean(node, goal):
    return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

# =============================
# A* 主循环
# =============================
def main(start, goal):
    while open_set:
        # 选择 f 最小的节点（正式决策）
        min_index = min(range(len(open_set)), key=lambda i: open_set[i][2])
        current = (open_set[min_index][0], open_set[min_index][1])

        # 确认采用
        open_set.pop(min_index)
        closed_set.append(current)
        visit_order.append(current)

        # 到达目标
        if current == goal:
            return current

        expand(current, goal)

    return None

def expand(current, goal):
    neighbors = [
        (current[0] + 1, current[1]),
        (current[0] - 1, current[1]),
        (current[0], current[1] + 1),
        (current[0], current[1] - 1),
    ]

    for neighbor in neighbors:
        if neighbor in closed_set:
            continue
        if neighbor in obstacle:
            continue
        if not (0 <= neighbor[0] <= grid_size[0] and
                0 <= neighbor[1] <= grid_size[1]):
            continue
        if neighbor in [(n[0], n[1]) for n in open_set]:
            continue

        g_cost[neighbor] = g_cost[current] + 1
        h_cost = Euclidean(neighbor, goal)
        f_cost = g_cost[neighbor] + h_cost

        # 仅加入候选集，不可视化
        open_set.append((neighbor[0], neighbor[1], f_cost, h_cost, g_cost[neighbor]))
        came_from[neighbor] = current

# =============================
# 回溯当前“已确认路径前缀”
# =============================
def reconstruct_partial_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# =============================
# 动画可视化（严格语义）
# =============================
def animate_confirmed_path(visit_order):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_xticks(range(grid_size[0] + 1))
    ax.set_yticks(range(grid_size[1] + 1))
    ax.grid(True, linestyle='--', alpha=0.6)

    # 障碍
    for obs in obstacle:
        ax.add_patch(plt.Rectangle(obs, 1, 1, color='dimgray'))

    # 起点 / 终点
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    # 已确认节点
    confirmed_scatter = ax.scatter([], [], c='skyblue', s=50, label='Confirmed')

    # 当前最优路径前缀
    path_line, = ax.plot([], [], c='navy', linewidth=2, label='Confirmed Path')

    def update(frame):
        current = visit_order[frame]

        # 已确认节点（closed_set 前缀）
        pts = visit_order[:frame + 1]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        confirmed_scatter.set_offsets(list(zip(xs, ys)))

        # 当前最优路径（start → current）
        path = reconstruct_partial_path(came_from, current)
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        path_line.set_data(px, py)

        return confirmed_scatter, path_line

    ani = FuncAnimation(
        fig,
        update,
        frames=len(visit_order),
        interval=300,
        repeat=False
    )

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("A* Confirmed-Path Online Visualization")
    plt.show()

# =============================
# 程序入口
# =============================
end = main(start, goal)

if end:
    animate_confirmed_path(visit_order)
else:
    print("未找到路径")
