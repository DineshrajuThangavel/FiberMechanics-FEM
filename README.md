🧮 Fiber Mechanics FEM – Plane Stress Simulation
This project implements a finite element solver in Python to model the plane stress response of orthotropic composite bars under axial loading. The solver incorporates fiber-angle-dependent stiffness transformation and visualizes the resulting anisotropic behavior. Developed as part of a computational mechanics course.

📌 Features
2D plane stress FEM using Q4 isoparametric elements

Fiber orientation via analytical stiffness rotation

4-point Gauss quadrature for element integration

Newton–Raphson method for global equilibrium solving

Analytical validation using compliance-based displacement

Mesh convergence study and fiber angle sweep analysis

🧾 Files
anisotropic_bar_fem.py: Main Python script implementing the FEM solver, visualization, and analysis routines
FiberMechanicsFEM_Report_2025.pdf: Full documentation with theory, implementation details, results, and plots

📊 Outputs
Structured mesh visualization

Tip displacement vs fiber angle (0°–360°)

Axial stress and strain plots

Effective modulus vs fiber orientation

Mesh refinement vs relative displacement error

🎓 Academic Context
Developed as part of the 2025 course on AI-Assisted Programming in Computational Mechanics at TU Freiberg, this project investigates the effect of fiber orientation on composite bar response using a fully custom FEM solver.

🔧 Technologies
Python (NumPy, Matplotlib)
FEM solver and postprocessing implemented entirely from scratch
