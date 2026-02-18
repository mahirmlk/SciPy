# SciPy for Python & AI/ML

A comprehensive collection of Jupyter notebooks covering SciPy topics essential for Python programming, Artificial Intelligence, and Machine Learning.

## 📚 Notebook Collection

| # | Notebook | Topics | Key Applications |
|---|----------|--------|------------------|
| 0 | [00_SciPy_Introduction.ipynb](./00_SciPy_Introduction.ipynb) | Overview, setup, basic examples | Getting started with SciPy |
| 1 | [01_Linear_Algebra.ipynb](./01_Linear_Algebra.ipynb) | Matrix decompositions, eigenvalues, SVD | PCA, neural networks, recommendation systems |
| 2 | [02_Optimization.ipynb](./02_Optimization.ipynb) | Minimization, curve fitting, root finding | Model training, hyperparameter tuning |
| 3 | [03_Statistics.ipynb](./03_Statistics.ipynb) | Distributions, hypothesis tests, descriptive stats | A/B testing, feature analysis, data exploration |
| 4 | [04_Signal_Processing.ipynb](./04_Signal_Processing.ipynb) | Filters, FFT, convolution | Time series, audio processing, feature extraction |
| 5 | [05_Interpolation.ipynb](./05_Interpolation.ipynb) | 1D/2D interpolation, splines, RBF | Missing data imputation, resampling |
| 6 | [06_Integration.ipynb](./06_Integration.ipynb) | Numerical integration, ODE solvers | Probability, Neural ODEs, physics simulations |
| 7 | [07_Sparse_Matrices.ipynb](./07_Sparse_Matrices.ipynb) | Sparse formats, operations, PageRank | NLP, recommendation systems, graph ML |
| 8 | [08_Spatial_Data.ipynb](./08_Spatial_Data.ipynb) | Distance metrics, KD-trees, Voronoi | KNN, clustering, nearest neighbor search |
| 9 | [09_Image_Processing.ipynb](./09_Image_Processing.ipynb) | Filters, morphology, measurements | Computer vision, medical imaging, segmentation |

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib jupyter
```

### Running the Notebooks

1. Clone or download this repository
2. Navigate to the folder containing the notebooks
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open any notebook and run cells sequentially

## 📖 Topics Covered

### 1. Linear Algebra (`scipy.linalg`)
- LU, QR, Cholesky decompositions
- Eigenvalue problems
- Singular Value Decomposition (SVD)
- Matrix norms and condition numbers
- **ML Applications**: PCA, linear regression, neural network layers

### 2. Optimization (`scipy.optimize`)
- Unconstrained minimization (BFGS, Nelder-Mead)
- Constrained optimization (SLSQP)
- Global optimization (differential evolution, simulated annealing)
- Curve fitting and root finding
- **ML Applications**: Model training, hyperparameter tuning, SVM

### 3. Statistics (`scipy.stats`)
- Probability distributions (continuous & discrete)
- Descriptive statistics
- Hypothesis testing (t-test, ANOVA, chi-square)
- Normality tests
- **ML Applications**: A/B testing, feature selection, data analysis

### 4. Signal Processing (`scipy.signal`)
- Digital filters (Butterworth, Chebyshev)
- FFT and spectral analysis
- Convolution operations
- Peak finding
- **ML Applications**: Time series analysis, audio features, CNNs

### 5. Interpolation (`scipy.interpolate`)
- 1D interpolation (linear, cubic, spline)
- 2D grid interpolation
- RBF interpolation for scattered data
- **ML Applications**: Missing data imputation, data augmentation

### 6. Integration (`scipy.integrate`)
- Numerical quadrature
- ODE solvers (RK45, BDF, LSODA)
- Multiple integration
- **ML Applications**: Probability calculations, Neural ODEs, physics-informed ML

### 7. Sparse Matrices (`scipy.sparse`)
- Sparse formats (CSR, CSC, COO, LIL)
- Sparse linear algebra
- PageRank algorithm
- **ML Applications**: NLP (TF-IDF), recommendation systems, graph neural networks

### 8. Spatial Data (`scipy.spatial`)
- Distance metrics
- KD-trees for nearest neighbors
- Voronoi diagrams
- Convex hull
- **ML Applications**: KNN, clustering, anomaly detection

### 9. Image Processing (`scipy.ndimage`)
- Gaussian filtering and denoising
- Edge detection (Sobel, Laplacian)
- Mathematical morphology
- Object measurements
- **ML Applications**: Computer vision preprocessing, medical image analysis

## 🎯 Real-World Examples

Each notebook contains multiple real-world examples:

| Domain | Examples |
|--------|----------|
| **Finance** | Portfolio optimization, risk analysis |
| **Healthcare** | ECG analysis, cell counting, epidemiological modeling |
| **E-commerce** | Recommendation systems, A/B testing |
| **IoT/Sensors** | Signal filtering, anomaly detection |
| **NLP** | Document-term matrices, TF-IDF vectorization |
| **Computer Vision** | Edge detection, image segmentation |

## 📊 SciPy Module Reference

```
scipy/
├── linalg          # Linear algebra operations
├── optimize        # Optimization algorithms
├── stats           # Statistical functions
├── signal          # Signal processing
├── interpolate     # Interpolation methods
├── integrate       # Integration and ODE solvers
├── sparse          # Sparse matrices
├── spatial         # Spatial algorithms
├── ndimage         # N-dimensional image processing
└── special         # Special mathematical functions
```

## 🔗 Additional Resources

- [SciPy Official Documentation](https://docs.scipy.org/doc/scipy/)
- [SciPy User Guide](https://docs.scipy.org/doc/scipy/tutorial/index.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Lecture Notes](https://scipy-lectures.org/)

## 📝 License

This collection is provided for educational purposes. Feel free to use and modify for your learning.

---

**Happy Learning! 🎓**
