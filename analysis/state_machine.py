from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_PATH = PROJECT_ROOT / "analysis" / "clusters.npy"
SAVE_TRANSITIONS = PROJECT_ROOT / "analysis" / "state_transitions.npy"
SAVE_FIG = PROJECT_ROOT / "analysis" / "plots" / "state_machine.png"


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

    # 更适合展示的布局：4 个状态时固定成菱形；否则退回到圆形
    if n_states == 4:
        pos = {
            0: (1.75, 0.0),
            1: (0.0, 1.75),
            2: (-1.75, 0.0),
            3: (0.0, -1.75),
        }
    else:
        pos = nx.circular_layout(G, scale=2.2)

    fig, ax = plt.subplots(figsize=(10.4, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.05, right=0.74, top=0.92, bottom=0.08)

    # 节点
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1450,
        node_color="#b9dff0",
        edgecolors="#6b8796",
        linewidths=1.0,
        ax=ax,
    )

    # 标签
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="semibold", ax=ax)

    def draw_edge(edge, rad, width, color, alpha=1.0):
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
            edge_color=color,
            alpha=alpha,
            ax=ax,
        )

    cross_style = {
        frozenset((1, 2)): (0.48, "#2b6cb0"),
        frozenset((1, 3)): (0.64, "#c05621"),
        frozenset((0, 3)): (0.52, "#2f855a"),
    }

    cross_edges = []
    for u, v in G.edges():
        weight = G[u][v]["weight"]
        if u == v:
            loop_rad = 0.20 + 0.04 * u
            width = min(0.9 + weight * 0.004, 1.8)
            draw_edge((u, v), loop_rad, width, "#94a3b8", alpha=0.45)
        else:
            base_rad, color = cross_style.get(frozenset((u, v)), (0.34, "#4a5568"))
            rad = base_rad if u < v else -base_rad
            width = min(1.15 + weight * 0.04, 2.4)
            draw_edge((u, v), rad, width, color, alpha=0.95)
            cross_edges.append((u, v, weight))

    summary_lines = ["Cross-state transition counts"]
    for u, v, weight in sorted(
        cross_edges, key=lambda item: (-item[2], item[0], item[1])
    ):
        summary_lines.append(f"{u} -> {v}: {weight}")
    summary_lines.append("Self-loops are shown as faint gray arcs")

    fig.text(
        0.77,
        0.50,
        "\n".join(summary_lines),
        ha="left",
        va="center",
        fontsize=9,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#cbd5e1", alpha=0.98),
    )

    ax.set_title("Symbolic State Machine", pad=14, fontsize=14)
    ax.axis("off")
    fig.savefig(save_fig, dpi=300)
    print(f"Saved {save_fig}")


enhanced_state_machine = build_state_machine


if __name__ == "__main__":
    build_state_machine()
