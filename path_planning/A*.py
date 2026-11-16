import matplotlib.pyplot as plt

start = (0,0)
open_set =[(start[0], start[1], 0, 0, 0)]
goal = (10,10)
grid_size = (20,20)
# obstacle = [(1,2),(2,2),(3,2),(4,2)]
obstacle = [(1,2),(2,2),(3,2),(4,2),(5,2),
            (3,0),(3,1),(3,3),(3,4),(3,5)]
# g = 0
current = start
g_cost = {start: 0}
# open_set.append(start)
closed_set = []
global came_from
came_from = {}

def main(start, goal):
    while open_set:
        min_index = min(range(len(open_set)), key=lambda i: open_set[i][2])
        print("open_set:", open_set)
        print("min_index:", min_index)
        current = (open_set[min_index][0], open_set[min_index][1])
        closed_set.append(current)
        print("当前节点：", current)
        if current[0] == goal[0] and current[1] == goal[1]:
            print("到达目标点")
            return current
        a_star(current, goal, grid_size)

def a_star(current, goal, grid_size=grid_size):
    for node in open_set[:]:
        if node[0]==current[0] and node[1]==current[1]:
            open_set.remove(node)
       
    neighbors_1 = (current[0]+1, current[1])
    neighbors_2 = (current[0]-1, current[1])
    neighbors_3 = (current[0], current[1]+1)
    neighbors_4 = (current[0], current[1]-1)
    neighbors = [neighbors_1, neighbors_2, neighbors_3, neighbors_4]
    print("邻居节点：", neighbors)
    for neighbor in neighbors:
        if neighbor in closed_set:
            continue
        if neighbor in obstacle:
            continue
        if neighbor[0]<0 or neighbor[1]<0 or neighbor[0]>grid_size[0] or neighbor[1]>grid_size[1]:
            continue
        if neighbor in [ (node[0], node[1]) for node in open_set]:
            continue
        h_cost = Euclidean(neighbor, goal)
        g_cost[neighbor] = g_cost[current] + 1
        f_cost = g_cost[neighbor]+ h_cost
        open_set.append((neighbor[0],neighbor[1], f_cost, h_cost, g_cost[neighbor]))
        came_from[neighbor] = current

def drawing(path=None, grid_size=(10, 10)):
    width, height = grid_size
    fig, ax = plt.subplots(figsize=(8, 8))

    # ========= 1. 坐标网格（参考用） =========
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks(range(width + 1))
    ax.set_yticks(range(height + 1))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    # ========= 2. 障碍物（占据区域） =========
    for obs in obstacle:
        ax.add_patch(
            plt.Rectangle(
                obs,
                1, 1,
                facecolor='dimgray',
                edgecolor='black',
                linewidth=0.8,
                zorder=2
            )
        )

    # ========= 3. 起点和终点（真实坐标点） =========
    ax.plot(
        start[0], start[1],
        marker='o',
        color='limegreen',
        markersize=10,
        label='Start',
        zorder=4
    )

    ax.plot(
        goal[0], goal[1],
        marker='o',
        color='crimson',
        markersize=10,
        label='Goal',
        zorder=4
    )

    # ========= 4. A* 路径（真实方向箭头） =========
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]

            dx = x1 - x0
            dy = y1 - y0

            ax.arrow(
                x0, y0,
                dx, dy,
                head_width=0.25,
                head_length=0.25,
                fc='navy',
                ec='navy',
                linewidth=2,
                length_includes_head=True,
                zorder=5
            )

    # ========= 5. 视觉设置 =========
    ax.set_aspect('equal')
    ax.set_title("A* Path Planning (Vertex-based Visualization)", fontsize=14)
    ax.legend(loc='upper left')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()




#曼哈顿距离启发函数
def Manhattan(node, goal):
    distance = abs(node[0]-goal[0]) + abs(node[1]-goal[1])
    return distance

def Euclidean(node, goal):
    return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

end = main(start, goal)
path = reconstruct_path(came_from, end)
drawing(path, grid_size)