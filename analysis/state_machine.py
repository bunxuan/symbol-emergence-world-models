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

    # 更清晰的布局
    pos = nx.spring_layout(G, seed=7, k=2.2, iterations=300)

    plt.figure(figsize=(8, 8))

    # 节点
    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color="lightblue")

    # 标签
    nx.draw_networkx_labels(G, pos, font_size=14)

    # 边（根据权重调整粗细，但做上限，避免自环过粗）
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    widths = [min(1.5 + w * 0.08, 3.0) for w in weights]
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=18,
        width=widths,
        connectionstyle="arc3,rad=0.12",
        min_source_margin=12,
        min_target_margin=12,
    )

    # 边标签
    edge_labels = {
        (i, j): T[i, j] for i in range(n_states) for j in range(n_states) if T[i, j] > 0
    }
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=11,
        label_pos=0.58,
        rotate=False,
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
