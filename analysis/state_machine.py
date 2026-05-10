from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_PATH = PROJECT_ROOT / "analysis" / "clusters.npy"
SAVE_TRANSITIONS = PROJECT_ROOT / "analysis" / "state_transitions.npy"
SAVE_FIG = PROJECT_ROOT / "results" / "state_machine.png"


def build_state_machine(
    cluster_path=CLUSTER_PATH,
    save_transitions=SAVE_TRANSITIONS,
    save_fig=SAVE_FIG,
):
    labels = np.load(cluster_path)
    transitions = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]
    transitions = np.array(transitions)

    np.save(save_transitions, transitions)
    print(f"Saved {save_transitions}")

    unique_states = np.unique(labels)
    n_states = len(unique_states)

    # 转移矩阵
    T = np.zeros((n_states, n_states), dtype=int)
    for a, b in transitions:
        T[a, b] += 1

    # 构建图
    G = nx.DiGraph()
    for s in unique_states:
        G.add_node(s)

    for i in range(n_states):
        for j in range(n_states):
            if T[i, j] > 0:
                G.add_edge(i, j, weight=T[i, j])

    # 更清晰的布局：固定成圆形，减少节点和边的挤压
    pos = nx.circular_layout(G, scale=2.2)

    plt.figure(figsize=(8, 8))

    # 节点
    nx.draw_networkx_nodes(G, pos, node_size=1300, node_color="lightblue")

    # 标签
    nx.draw_networkx_labels(G, pos, font_size=14)

    # 边（根据权重调整粗细，但做上限，避免自环过粗）
    edge_rads = {
        (0, 1): 0.18,
        (1, 0): -0.18,
        (1, 2): 0.18,
        (2, 1): -0.18,
        (2, 3): 0.18,
        (3, 2): -0.18,
        (3, 0): 0.18,
        (0, 3): -0.18,
        (0, 2): 0.34,
        (2, 0): -0.34,
        (1, 3): 0.34,
        (3, 1): -0.34,
    }

    def draw_edge(edge, rad, width):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            arrowstyle="-|>",
            arrowsize=18,
            width=width,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=12,
            min_target_margin=12,
        )

    # 先画非自环边，再单独画自环
    edge_labels = {}
    for u, v in G.edges():
        weight = G[u][v]["weight"]
        width = min(1.3 + weight * 0.06, 2.8)
        if u == v:
            loop_rad = 0.25 + 0.05 * u
            draw_edge((u, v), loop_rad, width)
        else:
            draw_edge((u, v), edge_rads.get((u, v), 0.16 if u < v else -0.16), width)
            edge_labels[(u, v)] = weight

    # 非自环边标签
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=11,
        label_pos=0.58,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
    )

    # 自环标签单独放，避免压在节点和环上
    for node in G.nodes():
        if T[node, node] > 0:
            x, y = pos[node]
            plt.text(
                x,
                y + 0.38,
                str(T[node, node]),
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )

    plt.title("Enhanced Symbolic State Machine")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    print(f"Saved {save_fig}")


enhanced_state_machine = build_state_machine


if __name__ == "__main__":
    build_state_machine()
