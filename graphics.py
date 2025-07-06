import matplotlib.pyplot as plt
import pandas as pd

def plot_bound_sol(solutions_list,cases):
    fig, axis = plt.subplots(1, 2, figsize=(12, 4))

    # First subplot
    plt.sca(axis[0])
    lines = []
    for idx, (alpha, beta) in enumerate(cases):
        sol = solutions_list[idx]
        line = sol.plot_overline(0, 0)
        lines.append(line[0])
        lines[-1].set_label(f"alpha={alpha:.2f}, beta={beta:.2f}")

    axis[0].legend()
    axis[0].margins(x=0.1, y=0.1)
    # Second subplot
    plt.sca(axis[1])
    lines = []
    for idx, (alpha, beta) in enumerate(cases):
        sol = solutions_list[idx]
        line = sol.plot_overline(3, 1)
        lines.append(line[0])
        lines[-1].set_label(f"alpha={alpha:.2f}, beta={beta:.2f}")

    axis[1].legend()
    axis[1].margins(x=0.1, y=0.1)
    plt.show()

def plot_table(methods, cases, list1, list2, variable_1, variable_2):

    # Prepare data with MultiIndex (alpha, beta) and variable (Time, Residual)
    data = []
    for i, (alpha, beta) in enumerate(cases):
        for method, t_list, n_list in zip(methods, list1, list2):
            data.append({
                'alpha': f"{alpha:.2f}",
                'beta': f"{beta:.2f}",
                'Method': method,
                'Variable': variable_1,
                'Value': f"{t_list[i]:.3f}"
            })
            data.append({
                'alpha': f"{alpha:.2f}",
                'beta': f"{beta:.2f}",
                'Method': method,
                'Variable': variable_2,
                'Value': f"{n_list[i]:.0f}"
            })

    df = pd.DataFrame(data)
    df.set_index(['alpha', 'beta', 'Variable'], inplace=True)
    table = df.pivot(columns='Method', values='Value')
    display(table)

def plot_combined(methods, res_histories, case_labels, cases):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Plot for case 2 (index 1)
    for method, res_list in zip(methods, res_histories):
        axes[0].plot(res_list[1], label=method)
    axes[0].set_title(f"Residual History - {case_labels[1]}")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Residual")
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True)

    # Plot for case 3 (index 2)
    for method, res_list in zip(methods, res_histories):
        axes[1].plot(res_list[2], label=method)
    axes[1].set_title(f"Residual History - {case_labels[2]}")
    axes[1].set_xlabel("Iteration")
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.tri as tri

def plot_solutions_grid(solutions_grid, cases, methods):
    """
    solutions_grid: 2D list/array of solution objects, shape (n_methods, n_cases)
    cases: list of (alpha, beta) tuples
    methods: list of method names (strings)
    """
    import matplotlib.tri as tri
    import matplotlib.pyplot as plt

    n_methods = len(methods)
    n_cases = len(cases)
    all_u = [sol.vect for row in solutions_grid for sol in row]
    vmin = min(u.min() for u in all_u)
    vmax = max(u.max() for u in all_u)

    fig, axes = plt.subplots(
        n_methods, n_cases, 
        figsize=(4*n_cases, 2.0*n_methods), 
        sharex=True, sharey=True,
        gridspec_kw={'wspace': 0.15, 'hspace': 0.05}
    )
    if n_methods == 1:
        axes = [axes]
    if n_cases == 1:
        axes = [[ax] for ax in axes]

    tpc = None
    for i, method in enumerate(methods):
        for j, (alpha, beta) in enumerate(cases):
            sol = solutions_grid[i][j]
            nodes = sol.nodes
            elements = sol.conn_matrix
            u = sol.vect
            triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
            ax = axes[i][j]
            tpc = ax.tripcolor(triangulation, u, shading='flat', cmap='viridis', vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_title(f'α={alpha:.2f}, β={beta:.2f}')
            if j == 0:
                ax.set_ylabel(method)
            ax.set_xlabel('X')
            ax.set_aspect('equal')
            ax.grid(True)

    # Adjust layout before adding colorbar
    plt.subplots_adjust(right=0.87, left=0.08, top=0.92, bottom=0.08, wspace=0.15, hspace=0.05)
    # Place colorbar to the right of all subplots
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    fig.colorbar(tpc, cax=cbar_ax, label='Solution value')
    plt.show()
