# 🧮 Fiber Mechanics FEM – Plane Stress Simulation

This project implements a finite element solver in Python to model the plane stress behavior of orthotropic composite bars under axial loading. The solver captures anisotropic effects by rotating the stiffness matrix based on fiber orientation. Developed as part of a computational mechanics course.

## 📌 Features
- 2D plane stress FEM using Q4 isoparametric elements
- Fiber-angle-dependent stiffness transformation
- 4-point Gauss quadrature for element integration
- Newton–Raphson method for solving global equilibrium
- Analytical comparison using compliance matrix solution
- Mesh convergence study and fiber angle sweep (0°–360°)

## 🧾 Files
- `anisotropic_bar_fem.py`: Main Python script implementing the FEM solver, fiber sweep, and plotting routines
- `FiberMechanicsFEM_Report_2025.pdf`: Full documentation with theoretical background, implementation, results, and validation

## 📊 Outputs
- Structured mesh visualization
- Tip displacement vs fiber angle
- Axial stress and strain plots
- Effective modulus variation with fiber direction
- Mesh convergence: relative error vs DOFs

## 🔧 Technologies
- Python (NumPy, Matplotlib)
- FEM solver and postprocessing implemented from scratch

## 🎓 Academic Context
Developed as part of the course on AI-Assisted Programming in Computational Mechanics.
