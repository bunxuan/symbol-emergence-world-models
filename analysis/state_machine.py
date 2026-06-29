from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_PATH = PROJECT_ROOT / "analysis" / "clusters.npy"
SAVE_TRANSITIONS = PROJECT_ROOT / "analysis" / "state_transitions.npy"
SAVE_FIG = PROJECT_ROOT / "analysis" / "plots" / "state_machine.png"
SAVE_MI_STATS = PROJECT_ROOT / "analysis" / "symbol_mutual_info.npz"
SAVE_MI_FIG = (
    PROJECT_ROOT
    / "report"
    / "figures"
    / "entropy"
    / "fig9c_symbol_mutual_information.png"
)


def build_state_machine(
    cluster_path=CLUSTER_PATH,
    save_transitions=SAVE_TRANSITIONS,
    save_fig=SAVE_FIG,
):
    labels = np.load(cluster_path)
    transitions = [(labels[i], labels[i + 1]) for i in range(len(labels) - 1)]
    transitions = np.array(transitions)
    symbol_mi = float(mutual_info_score(labels[:-1], labels[1:]))
    symbol_nmi = float(normalized_mutual_info_score(labels[:-1], labels[1:]))

    SAVE_MI_STATS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        SAVE_MI_STATS,
        mutual_information=symbol_mi,
        normalized_mutual_information=symbol_nmi,
    )
    print(f"Saved {SAVE_MI_STATS}")

    SAVE_MI_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig_mi, ax_mi = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    metric_names = ["I(S_t;S_{t+1})", "NMI"]
    metric_values = [symbol_mi, symbol_nmi]
    metric_colors = ["#2563eb", "#d97706"]
    ax_mi.bar(metric_names, metric_values, color=metric_colors, edgecolor="#1f2937")
    ax_mi.set_ylabel("score")
    ax_mi.set_title("Symbol mutual information")
    ax_mi.set_ylim(0.0, max(1.0, max(metric_values) * 1.2))
    ax_mi.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(metric_values):
        ax_mi.text(
            idx, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=9
        )
    ax_mi.text(
        0.02,
        0.98,
        "Discrete symbols retain predictive dependence\nbetween adjacent time steps.",
        transform=ax_mi.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.95),
    )
    fig_mi.savefig(SAVE_MI_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig_mi)
    print(f"Saved {SAVE_MI_FIG}")

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

    summary_lines = [
        f"I(S_t; S_t+1) = {symbol_mi:.3f} nats",
        f"NMI = {symbol_nmi:.3f}",
        "Cross-state transition counts",
    ]
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
