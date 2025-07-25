📐 FiberMechanics-FEM – Plane Stress Simulation
A Python-based finite element solver for analyzing the plane stress response of orthotropic composite bars under axial loading. Captures directional anisotropy through fiber-angle-dependent stiffness transformation and compares FEM results with analytical predictions. Developed as part of AI-Assisted Programming in Computational Mechanics (2025, TU Freiberg).

🚀 Key Features
✅ Plane stress FEM using Q4 isoparametric elements

✅ Rotated stiffness matrix for arbitrary fiber orientation

✅ 4-point Gauss integration for element stiffness

✅ Newton–Raphson method for global system solving

✅ Analytical comparison using compliance theory

✅ Mesh convergence with relative error analysis

✅ Full sweep of fiber angles (0°–360°)

✅ Displacement, stress, strain, and modulus visualization

📂 Included Files
File	Description
anisotropic_bar_fem.py	Main Python script with FEM solver, postprocessing, and plots
FiberMechanicsFEM_Report_2025.pdf	Final report detailing theory, implementation, and results

📈 Generated Outputs
Structured mesh visualization

Tip displacement vs. fiber angle

Stress (σₓₓ) and strain (εₓₓ) plots

Effective axial modulus vs. fiber angle

Mesh convergence plot: relative error vs. DOFs

🧪 How to Run
Ensure Python 3.x is installed with the following libraries:

numpy

matplotlib

Then run the solver with:

bash
Copy
Edit
python anisotropic_bar_fem.py
🎓 Academic Context
This project was completed for the AI-Assisted Programming in Computational Mechanics (2025) course at TU Freiberg. It investigates fiber orientation effects on the axial response of anisotropic composites using FEM techniques implemented from scratch in Python.
