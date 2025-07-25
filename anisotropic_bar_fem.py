import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. Global font and style settings
mpl.rc('font',      size=10)    # main font size
mpl.rc('axes',      titlesize=11, labelsize=10)
mpl.rc('legend',    fontsize=10)
mpl.rc('xtick',     labelsize=9)
mpl.rc('ytick',     labelsize=9)
mpl.rc('figure',    figsize=(6, 3)) 

# --- Input Material Properties ---
Ef = 235e9  # Fiber modulus [Pa]
Em = 3.5e9  # Matrix modulus [Pa]
vf = 0.2
vm = 0.35
G12 = 5e9
Vf = 0.572
Vm = 0.428
L = 1.0
height = 0.05  # m
A = height * 1.0  # (m^2, for 1m width in 2D)
nx, ny = 50, 10  # mesh

# --- Material Properties Calculation ---
E1 = Ef * Vf + Em * Vm
E2 = 1 / (Vf / Ef + Vm / Em)
v12 = vf * Vf + vm * Vm
v21 = v12 * E2 / E1

denom = 1 - v12 * v21
Q11 = E1 / denom
Q22 = E2 / denom
Q12 = v12 * E2 / denom
Q66 = G12
Q = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

def transform_stiffness(Q, theta_deg):
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    Q11, Q12, _, Q22, _, Q66 = Q[0,0], Q[0,1], Q[0,2], Q[1,1], Q[1,2], Q[2,2]
    # Analytical transformation
    Q11_star = Q11 * c**4 + 2 * (Q12 + 2*Q66) * s**2 * c**2 + Q22 * s**4
    Q22_star = Q11 * s**4 + 2 * (Q12 + 2*Q66) * s**2 * c**2 + Q22 * c**4
    Q12_star = (Q11 + Q22 - 4*Q66) * s**2 * c**2 + Q12 * (s**4 + c**4)
    Q66_star = (Q11 + Q22 - 2*Q12 - 2*Q66) * s**2 * c**2 + Q66 * (s**4 + c**4)
    Qstar = np.array([[Q11_star, Q12_star, 0],
                      [Q12_star, Q22_star, 0],
                      [0, 0, Q66_star]])
    return Qstar

def create_mesh(length=1.0, height=0.05, nx=50, ny=10):
    dx, dy = length / nx, height / ny
    nodes, node_id_map = [], {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            node_id = j * (nx + 1) + i
            node_id_map[(i, j)] = node_id
            nodes.append([i * dx, j * dy])
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_id_map[(i, j)]
            n2 = node_id_map[(i + 1, j)]
            n3 = node_id_map[(i + 1, j + 1)]
            n4 = node_id_map[(i, j + 1)]
            elements.append([n1, n2, n3, n4])
    return np.array(nodes), np.array(elements)

def shape_functions(xi, eta):
    return 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [ (1 - eta), -(1 + xi)],
        [ (1 + eta),  (1 + xi)],
        [-(1 + eta),  (1 - xi)]
    ])

def calculate_stiffness_matrix(Q, coords):
    ke = np.zeros((8, 8))
    gauss_pts = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                 (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
    for xi, eta in gauss_pts:
        dN_dxi = shape_functions(xi, eta)
        J = sum(np.outer(dN_dxi[i], coords[i]) for i in range(4))
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Jacobian determinant is non-positive!")
        J_inv = np.linalg.inv(J)
        dN_dx = dN_dxi @ J_inv
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        ke += B.T @ Q @ B * detJ
    return ke

def assemble_global_stiffness(nodes, elements, Q_theta):
    K = np.zeros((2 * len(nodes), 2 * len(nodes)))
    for elem in elements:
        coords = nodes[elem]
        ke = calculate_stiffness_matrix(Q_theta, coords)
        dof_map = np.hstack([[2*n, 2*n+1] for n in elem])
        for i in range(8):
            for j in range(8):
                K[dof_map[i], dof_map[j]] += ke[i, j]
    return K

def apply_boundary_conditions(K, F, nodes):
    fixed_dofs = []
    for i, (x, y) in enumerate(nodes):
        if np.isclose(x, 0.0):
            fixed_dofs.extend([2*i, 2*i+1])
    free_dofs = np.setdiff1d(np.arange(len(F)), fixed_dofs)
    K_reduced = K[np.ix_(free_dofs, free_dofs)]
    F_reduced = F[free_dofs]
    return K_reduced, F_reduced, fixed_dofs, free_dofs

def apply_load(nodes, total_load, direction='x'):
    F = np.zeros(len(nodes)*2)
    right_edge_nodes = [i for i, node in enumerate(nodes) if np.isclose(node[0], L)]
    load_per_node = total_load / len(right_edge_nodes)
    for i in right_edge_nodes:
        idx = 2*i if direction=='x' else 2*i+1
        F[idx] = load_per_node
    return F, right_edge_nodes

def solve_for_displacements_newton(K_full, F_ext, nodes, elements, Q_theta, fixed_dofs, free_dofs,
                                    max_iters=25, tol=1e-6):
    U = np.zeros_like(F_ext)
    for step in range(max_iters):
        F_int = np.zeros_like(F_ext)
        K_tangent = np.zeros_like(K_full)
        for elem in elements:
            coords = nodes[elem]
            u_elem = np.hstack([U[2*n:2*n+2] for n in elem])
            gauss_pts = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                         (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
            f_int_local = np.zeros(8)
            ke_local = np.zeros((8, 8))
            for xi, eta in gauss_pts:
                dN_dxi = shape_functions(xi, eta)
                J = sum(np.outer(dN_dxi[i], coords[i]) for i in range(4))
                detJ = np.linalg.det(J)
                J_inv = np.linalg.inv(J)
                dN_dx = dN_dxi @ J_inv
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i]   = dN_dx[i, 0]
                    B[1, 2*i+1] = dN_dx[i, 1]
                    B[2, 2*i]   = dN_dx[i, 1]
                    B[2, 2*i+1] = dN_dx[i, 0]
                strain = B @ u_elem
                stress = Q_theta @ strain
                f_int_local += B.T @ stress * detJ
                ke_local += B.T @ Q_theta @ B * detJ
            dof_map = np.hstack([[2*n, 2*n+1] for n in elem])
            for i in range(8):
                F_int[dof_map[i]] += f_int_local[i]
                for j in range(8):
                    K_tangent[dof_map[i], dof_map[j]] += ke_local[i, j]
        R = F_ext - F_int
        R_reduced = R[free_dofs]
        K_reduced = K_tangent[np.ix_(free_dofs, free_dofs)]
        delta_U = np.linalg.solve(K_reduced, R_reduced)
        U[free_dofs] += delta_U
        norm_R = np.linalg.norm(R_reduced)
        norm_dU = np.linalg.norm(delta_U)
        print(f"[Newton Step {step}] ||R|| = {norm_R:.2e}, ||ΔU|| = {norm_dU:.2e}")
        if norm_R < tol:
            print("✅ Converged.")
            break
    else:
        print("⚠️ Newton–Raphson did not converge in max iterations.")
    return U

def compute_element_strain_stress(nodes, elements, U, Q_theta):
    strain_list, stress_list = [], []
    for elem in elements:
        coords = nodes[elem]
        u_elem = np.hstack([U[2*n:2*n+2] for n in elem])
        elem_strains, elem_stresses = [], []
        gauss_pts = [(-1/np.sqrt(3), -1/np.sqrt(3)), (1/np.sqrt(3), -1/np.sqrt(3)),
                     (1/np.sqrt(3), 1/np.sqrt(3)), (-1/np.sqrt(3), 1/np.sqrt(3))]
        for xi, eta in gauss_pts:
            dN_dxi = shape_functions(xi, eta)
            J = sum(np.outer(dN_dxi[i], coords[i]) for i in range(4))
            J_inv = np.linalg.inv(J)
            dN_dx = dN_dxi @ J_inv
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i]   = dN_dx[i, 0]
                B[1, 2*i+1] = dN_dx[i, 1]
                B[2, 2*i]   = dN_dx[i, 1]
                B[2, 2*i+1] = dN_dx[i, 0]
            strain = B @ u_elem
            stress = Q_theta @ strain
            elem_strains.append(strain)
            elem_stresses.append(stress)
        strain_list.append(np.mean(elem_strains, axis=0))
        stress_list.append(np.mean(elem_stresses, axis=0))
    return np.array(strain_list), np.array(stress_list), elements


def analytical_tip_displacement_full_2D(total_load, L, height, Q, theta_deg, y_tip=height):
    """
    Analytical tip x- and y-displacement at the tip, using full 2D compliance matrix.
    y_tip: vertical location at tip (0.0 for bottom tip, height for top tip)
    """
    A = height * 1.0
    sigma = np.array([total_load / A, 0.0, 0.0])
    Qstar = transform_stiffness(Q, theta_deg)
    Sstar = np.linalg.inv(Qstar)
    strain = Sstar @ sigma
    ux_tip = strain[0] * L
    uy_tip = strain[1] * y_tip
    return ux_tip, uy_tip, strain


def fem_tip_displacement(theta, total_load):
    Q_theta = transform_stiffness(Q, theta)
    nodes, elements = create_mesh(length=L, height=height, nx=nx, ny=ny)
    K = assemble_global_stiffness(nodes, elements, Q_theta)
    F, right_edge_nodes = apply_load(nodes, total_load, direction='x')
    K_reduced, F_reduced, fixed_dofs, free_dofs = apply_boundary_conditions(K, F, nodes)
    U = solve_for_displacements_newton(K, F, nodes, elements, Q_theta, fixed_dofs, free_dofs)
    # Tip node index (x=1.0, y=height)
    tip_indices = np.where(np.isclose(nodes[:,0], L) & np.isclose(nodes[:,1], height))[0]
    Ux_tip = U[2*tip_indices[0]] if len(tip_indices) > 0 else np.nan
    Uy_tip = U[2*tip_indices[0]+1] if len(tip_indices) > 0 else np.nan
    return Ux_tip, Uy_tip, U, nodes, elements

def run_case(theta, total_load):
    print(f"--- Orientation: {theta} degrees ---")
    Ux_tip, Uy_tip, U, nodes, elements = fem_tip_displacement(theta, total_load)
    strain, stress, _ = compute_element_strain_stress(nodes, elements, U, transform_stiffness(Q, theta))
    sigma_xx = stress[:, 0]
    # Analytical: use full 2D compliance, top tip (y_tip=height)
    Ux_ana, Uy_ana, strain_ana = analytical_tip_displacement_full_2D(total_load, L, height, Q, theta, y_tip=height)
    print(f"Tip Ux (FEM): {Ux_tip:.4e} m")
    print(f"Tip Ux (Analytical): {Ux_ana:.4e} m")
    print(f"Relative error (Ux): {100.0 * abs(Ux_tip - Ux_ana)/abs(Ux_ana):.2f}%")


def sweep_theta_cases(total_load):
    thetas = np.arange(0, 361, 5)
    Ux_fem, Uy_fem = [], []
    Ux_analytical, Uy_analytical = [], []
    avg_sigma_fem = []
    avg_sigma_ana = []
    avg_eps_fem = []
    avg_eps_ana = []
    E_theta_list = []
    E_theta_num = []
    for theta in thetas:
        # FEM
        ux_tip, uy_tip, U, nodes, elements = fem_tip_displacement(theta, total_load)
        strain, stress, _ = compute_element_strain_stress(nodes, elements, U, transform_stiffness(Q, theta))
        sigma_xx = stress[:, 0]
        epsilon_xx = strain[:, 0]
        Ux_fem.append(ux_tip)
        Uy_fem.append(uy_tip)
        avg_sigma_fem.append(np.mean(sigma_xx))
        avg_eps_fem.append(np.mean(epsilon_xx))
        # Numerical modulus: (max σₓₓ)/(avg εₓₓ)
        if np.mean(epsilon_xx) != 0:
            E_theta_num.append(np.mean(sigma_xx) / np.mean(epsilon_xx))
        else:
            E_theta_num.append(np.nan)
        # Analytical (bottom tip, y_tip = height)
        ux_ana, uy_ana, strain_ana = analytical_tip_displacement_full_2D(
            total_load, L, height, Q, theta, y_tip=height
        )
        Ux_analytical.append(ux_ana)
        Uy_analytical.append(uy_ana)
        # Now, for the analytical stress, calculate using Q* and strain (not just F/A)
        Qstar = transform_stiffness(Q, theta)
        sigma_ana = Qstar @ strain_ana  # [σ_xx, σ_yy, τ_xy]
        sigma_xx_ana = sigma_ana[0]
        avg_sigma_ana.append(sigma_xx_ana)
        eps_ana = strain_ana[0]
        avg_eps_ana.append(eps_ana)
        # Analytical modulus: σ_xx / ε_xx (full 2D response)
        E_theta_list.append(sigma_xx_ana / eps_ana if eps_ana != 0 else np.nan)
    # Convert all to numpy arrays
    return (thetas, np.array(Ux_fem), np.array(Ux_analytical), 
            np.array(Uy_fem), np.array(Uy_analytical),
            np.array(avg_sigma_fem), np.array(avg_sigma_ana),
            np.array(avg_eps_fem), np.array(avg_eps_ana),
            np.array(E_theta_num), np.array(E_theta_list))


def plot_bar_mesh(nodes, elements, nx, ny, title=None, color='black', lw=0.5):
    """
    Clean mesh visualization: draws all element edges in specified color.
    nodes: (N_nodes, 2), elements: (N_elem, 4) node-indices
    """
    plt.figure(figsize=(7.5, 3))  
    for elem in elements:
        xy = nodes[elem]
        xy_closed = np.vstack([xy, xy[0]])
        plt.plot(xy_closed[:,0], xy_closed[:,1],
                 color=color, linewidth=lw)
    plt.gca().set_aspect('auto')
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title(title or f"Structured mesh ({nx}×{ny})")
    plt.tight_layout()
    plt.show()

def plot_displacement_vs_theta(thetas, Ux_fem, Ux_ana):
    plt.figure(figsize=(7.5, 3))  
    plt.plot(thetas, Ux_fem,    '-o',  markersize=4,
             label='FEM tip $u_x$',      linewidth=1)
    plt.plot(thetas, Ux_ana,    '--s', markersize=4,
             label='Analytical $u_x$', linewidth=1)
    plt.xlabel("Fiber orientation θ (°)")
    plt.ylabel("Tip displacement $u_x$ (m)")
    plt.title("Tip displacement vs. fiber orientation")
    plt.grid(True, linestyle=':')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=(0, 0, 0.8, 1.0))
    plt.show()

def plot_sigma_xx_vs_theta(thetas, avg_sigma_fem, avg_sigma_ana):
    # 1. Create a new figure (6×3 in)
    plt.figure(figsize=(7.5, 3))  

    # 2. Plot FEM stress with small circles
    plt.plot(
        thetas,
        avg_sigma_fem / 1e6,
        '-o',
        markersize=3,
        linewidth=1,
        label='FEM avg $\\sigma_{xx}$'
    )
    # 3. Plot analytical stress with dashed line only
    plt.plot(
        thetas,
        avg_sigma_ana / 1e6,
        '--',
        linewidth=1,
        label='Analytical avg $\\sigma_{xx}$'
    )
    plt.xlabel("Fiber orientation $\\theta$ (°)")
    plt.ylabel("Average $\\sigma_{xx}$ (MPa)")
    plt.title("Average $\\sigma_{xx}$ vs. fiber orientation")
    plt.grid(True, linestyle=':')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=(0, 0, 0.8, 1.0))
    plt.show()
    
def plot_epsilon_xx_vs_theta(thetas, avg_eps_fem, avg_eps_ana):
    plt.figure(figsize=(7.5, 3))  
    plt.plot(thetas, avg_eps_fem, '-o', markersize=4,
             label='FEM avg εₓₓ', linewidth=1)
    plt.plot(thetas, avg_eps_ana, '--s', markersize=4,
             label='Analytical avg εₓₓ', linewidth=1)
    plt.xlabel("Fiber orientation θ (°)")
    plt.ylabel("Average εₓₓ (–)")   # dimensionless strain in round brackets
    plt.title("Average εₓₓ vs. fiber orientation")
    plt.grid(True, linestyle=':')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=(0, 0, 0.8, 1.0))
    plt.show()

def plot_modulus_vs_theta(thetas, E_theta_num, E_theta_ana):
    plt.figure(figsize=(7.5, 3))  
    plt.plot(thetas, E_theta_num/1e9, '-o', markersize=4,
             label=r'FEM $E_\theta$', linewidth=1)
    plt.plot(thetas, E_theta_ana/1e9, '--s', markersize=4,
             label=r'Analytical $E_\theta$', linewidth=1)
    plt.xlabel("Fiber orientation θ (°)")
    plt.ylabel(r"Effective modulus $E_\theta$ (GPa)")
    plt.title(r"Effective modulus $E_\theta$ vs. fiber orientation")
    plt.grid(True, linestyle=':')
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=(0, 0, 0.8, 1.0))
    plt.show()

def mesh_convergence_study(mesh_sizes, theta_deg, total_load, L, height, Q):
    """
    Runs FEM for different meshes and plots relative error in tip displacement.
    """
    tip_disp_num = []
    labels       = []
    
    # 1) Get analytical reference once
    ux_ana, _, _ = analytical_tip_displacement_full_2D(
        total_load, L, height, Q, theta_deg, y_tip=height
    )
    
    # 2) Loop over all meshes
    for nx_i, ny_i in mesh_sizes:
        # update global mesh parameters if your FEM uses them
        global nx, ny
        nx, ny = nx_i, ny_i
        
        # run your FEM solver
        ux_tip_num, _, _, _, _ = fem_tip_displacement(theta_deg, total_load)
        tip_disp_num.append(ux_tip_num)
        labels.append(f"{nx_i}×{ny_i}")
    
    # 3) Compute relative error (%) for each mesh
    rel_err = [100 * abs(u_num - ux_ana) / ux_ana
               for u_num in tip_disp_num]
    
    # 4) Plot convergence
    plt.figure(figsize=(7.5, 3))  
    plt.plot(labels, rel_err, '-o', markersize=4, linewidth=1)
    plt.xlabel("Mesh size $n_x\\times n_y$")
    plt.ylabel("Relative error in tip $u_x$ ( %)")
    plt.title("Mesh convergence: relative error in tip displacement")
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()




# --- Main organized execution ---
if __name__ == "__main__":
    # 1. Generate the structured mesh and visualize it (to show the geometry and mesh resolution)
    nodes, elements = create_mesh(length=L, height=height, nx=nx, ny=ny)
    plot_bar_mesh(nodes, elements, nx, ny, color='red', lw=1.2)  # Clean mesh style visualization

    # 2. Run and visualize results for specific fiber orientations (quick sanity check / illustration)
    for theta in [0, 45, 90]:
        run_case(theta, total_load=1000.0)

    # 3. Sweep all fiber orientations, collect FEM and analytical results for all metrics
    (thetas, Ux_fem, Ux_ana, Uy_fem, Uy_ana,
     avg_sigma_fem, avg_sigma_ana,
     avg_eps_fem, avg_eps_ana,
     E_theta_num, E_theta_ana) = sweep_theta_cases(total_load=1000.0)

    # 4. Plot key quantities versus fiber angle (θ): displacements, stress, strain, modulus
    plot_displacement_vs_theta(thetas, Ux_fem, Ux_ana)
    plot_sigma_xx_vs_theta(thetas, avg_sigma_fem, avg_sigma_ana)
    plot_epsilon_xx_vs_theta(thetas, avg_eps_fem, avg_eps_ana)
    plot_modulus_vs_theta(thetas, E_theta_num, E_theta_ana)

    # 5. Perform mesh convergence study (see how FEM tip displacement approaches analytical value)
    mesh_sizes = [(10, 2), (20, 4), (40, 8), (80, 16), (160, 32)]
    theta_deg = 45
    total_load = 1000.0
    mesh_convergence_study(mesh_sizes, theta_deg, total_load, L, height, Q)