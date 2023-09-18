# ReviewOfMultiFidelityModels_ToyProblems
The repository, titled ReviewOfMultiFidelityModels_ToyProblems, serves as an invaluable supplement to the academic journal article "Review of Multifidelity Models." Designed as a comprehensive collection of Jupyter Notebooks, the repository aims to present a tangible and easily understandable demonstration of multifidelity models. These models are showcased through a series of Python-based toy problems, thus facilitating a deeper understanding of the article's theoretical concepts. The repository contributes to the broader scientific understanding of multifidelity approaches by providing hands-on, executable examples.
# Multi-Fidelity Modeling Toy Problem 1: Additive and Multiplicative Corrections

This code provides a demonstration of multi-fidelity modeling techniques for refining low-fidelity (LF) models using high-fidelity (HF) data through additive and multiplicative corrections. The example uses one-dimensional analytic functions to illustrate these concepts and includes plotting for visualization.

## Code Structure

The code is organized into two main classes: `AddMultCorrPlotter` and `AddMultCorrModels`.

### `AddMultCorrPlotter` Class

This class is responsible for plotting LF, HF, and correction models.

#### Methods

- `func_HF(x)`: High-Fidelity function definition.
- `func_LF(x)`: Low-Fidelity function definition.
- `plot_functions()`: Plots LF and HF functions.
- `x`: X-axis values for plotting LF and HF functions.

### `AddMultCorrModels` Class

This class handles the modeling and plotting of additive and multiplicative corrections.

#### Methods

- `polynomial_fit(x, y, deg=2)`: Fits a polynomial model to data.
- `func_add(x)`: Computes the additive correction function.
- `func_mult(x)`: Computes the multiplicative correction function.
- `plot_additive_correction()`: Plots the additive correction.
- `plot_multiplicative_correction()`: Plots the multiplicative correction.
- `plot_combined_corrections()`: Plots combined corrections.

## Usage

1. Instantiate the `AddMultCorrPlotter` class to define LF and HF functions and plot them.
2. Instantiate the `AddMultCorrModels` class, passing the `AddMultCorrPlotter` instance.
3. Use the methods in the `AddMultCorrModels` class to plot additive and multiplicative corrections as needed.

## Example

```python
if __name__ == "__main__":
    plotter = AddMultCorrPlotter()
    plotter.plot_functions()
    
    models = AddMultCorrModels(plotter)
    
    plt.figure(figsize=[5, 3])
    models.plot_additive_correction()
    
    plt.figure(figsize=[5, 3])
    models.plot_multiplicative_correction()
    
    plt.figure(figsize=[5, 3.5])
    models.plot_combined_corrections()
    
    plt.show()
```

    
# Multi-Fidelity Modeling Toy Problem 2: Comprehensive Correction

## Overview
The presented code is a Python implementation designed to visualize High-Fidelity (HF) and Low-Fidelity (LF) functions and to conduct sensitivity analysis. The code leverages the Matplotlib and NumPy libraries for plotting and numerical operations, respectively. It encapsulates functionality into three classes: `FunctionVisualizer`, `ComprehensiveFunctionVisualizer`, and `SensitivityAnalysis`.

---

### FunctionVisualizer Class

#### Attributes
- **lb**: Lower bound of the domain for function visualization.
- **ub**: Upper bound of the domain for function visualization.
- **x**: NumPy array containing equidistant points within [lb, ub] at which function evaluations are carried out.

#### Methods
- **\_\_init\_\_(lb, ub)**: Initializes an object with specified lower and upper bounds for the domain.
- **func_HF(x)**: Evaluates the High-Fidelity function (HF) at a given point \( x \).
- **func_LF(x)**: Evaluates the Low-Fidelity function (LF) at a given point \( x \).
- **visualize()**: Generates a plot of both HF and LF functions within the specified domain.

---

### ComprehensiveFunctionVisualizer Class

#### Attributes
- **x_sampHF**: NumPy array representing the points at which the HF function is sampled.

#### Methods
- Inherits all methods from `FunctionVisualizer`.
- **func_X(x)**: Transforms the input \( x \) into a design matrix \( X \) by considering additional features.
- **func_coef(X, Y, W=1)**: Computes the model coefficients using weighted least squares.
- **func_comprehensive(x)**: Constructs a comprehensive model by combining HF and LF information.
- **visualize_comprehensive()**: Generates a plot incorporating HF, LF, and the comprehensive model along with HF sampling points.

---

### SensitivityAnalysis Class

#### Attributes
- **coef**: Dictionary containing coefficients 'A', 'B', and 'C' for LF function adjustments.

#### Methods
- Inherits the relevant methods for HF and LF functions, comprehensive model, and weighted least squares from previous classes.
- **plot_sensitivity(x, x_sampHF, coef_key, coef_val_range, plot_title)**: Generates sensitivity plots to analyze the impact of changing individual coefficients within specified ranges.

---

## Example Usage

### Function Visualization
```python
visualizer = FunctionVisualizer(0, 1)
visualizer.visualize()
```

### Comprehensive Modeling
```python
visualizer = ComprehensiveFunctionVisualizer(0, 1, np.array([0.1, 0.5, 0.9]))
visualizer.visualize_comprehensive()
```

### Sensitivity Analysis
```python
x = np.linspace(0, 1, 100)
x_sampHF = np.array([0.1, 0.5, 0.9])
sensitivity = SensitivityAnalysis()
sensitivity.plot_sensitivity(x, x_sampHF, 'A', np.linspace(-1.3, 2.7, 10), 'A_Sensitivity')
```
# Multi-Fidelity Modeling Toy Problem 3: MultiFidelityModel Class Documentation

## Overview

The `MultiFidelityModel` class encapsulates the functionality for generating a Multi-Fidelity Surrogate Model (MFSM). It aims to synergize high-fidelity model (HFM) and low-fidelity model (LFM) data to create a more accurate and computationally efficient model.

## Dependencies

- matplotlib.pyplot: Used for plotting.
- numpy: For numerical operations.
- sklearn.svm.SVR: For Support Vector Regression.
- sklearn.ensemble.RandomForestRegressor: For Random Forest Regression.

## Class Attributes

- `x`: Numpy array consisting of 100 points linearly spaced between 0 and 1.
- `nLF`: Integer, number of LFM samples.
- `nHF`: Integer, number of HFM samples.
- `svr`: Object of class `SVR`, used for the Support Vector Regression model.
- `rf`: Object of class `RandomForestRegressor`, used for the Random Forest model.

### Methods

#### `__init__(self, x, nLF, nHF)`

Initializes class instance variables.

- **Parameters:**
  - `nLF`: Number of Low-Fidelity Model samples.
  - `nHF`: Number of High-Fidelity Model samples.

#### `func_HF(x)`

Static method representing the high-fidelity function.

- **Parameters:**
  - `x`: Input variable values (Numpy ndarray).
- **Returns:**
  - Output response values (Numpy ndarray).

#### `func_LF(x)`

Static method representing the low-fidelity function.

- **Parameters:**
  - `x`: Input variable values (Numpy ndarray).
- **Returns:**
  - Output response values (Numpy ndarray).

#### `generate_data(self)`

Generates random sampling points for LFM and HFM and calculates corresponding function values.

#### `fit_SVR(self)`

Fits the SVR model to approximate the discrepancy between HFM and LFM.

#### `fit_RF(self)`

Fits the Random Forest model to the low-fidelity data, thus building the Low-Fidelity Surrogate Model (LFSM).

#### `y_MFSM_A(self, x)`

Calculates the integrated surrogate model using Option A.

- **Parameters:**
  - `x`: Input variable values (Numpy ndarray).
- **Returns:**
  - Output response values (Numpy ndarray).

#### `y_MFSM_B(self, x)`

Calculates the integrated surrogate model using Option B.

- **Parameters:**
  - `x`: Input variable values (Numpy ndarray).
- **Returns:**
  - Output response values (Numpy ndarray).

#### `plot(self)`

Generates a plot to visualize the actual LFM, HFM, and MFSM estimates. The plot also displays the discrepancy between the models.

## Example Usage

```python
if __name__ == "__main__":
    model = MultiFidelityModel(x=1000, nLF=200, nHF=20)
    model.generate_data()
    model.fit_SVR()
    model.fit_RF()
    model.plot()
```
# Multi-Fidelity Modeling Toy Problem 4: Multi-Fidelity Analysis Using Co-Kriging

### Introduction

The `MultiFidelityAnalysis` class is an encapsulation for performing multi-fidelity analysis via Co-Kriging (CoKG). The class provides methods to fit a CoKG model, make predictions, and visualize the Low-Fidelity (LF) and High-Fidelity (HF) models alongside CoKG predictions.

### Requirements

- Python 3.6 or higher
- NumPy
- Matplotlib
- OpenMDAO

### Installation

```bash
pip install numpy matplotlib openmdao
```

### Class Methods and Attributes

#### `__init__(self, lb=0, ub=1)`

Initializes the `MultiFidelityAnalysis` object. 

**Parameters:**

- `lb`: Lower bound of the variable space (default is 0).
- `ub`: Upper bound of the variable space (default is 1).

#### `func_HF(x)`

High-Fidelity (HF) model as a function of `x`.

**Returns:**

- HF model response.

#### `func_LF(x)`

Low-Fidelity (LF) model as a function of `x`.

**Returns:**

- LF model response.

#### `fit_coKG(self, Xe, Xc)`

Fits the CoKG model with HF and LF data.

**Parameters:**

- `Xe`: Points where the HF model is evaluated.
- `Xc`: Points where the LF model is evaluated.

#### `predict_coKG()`

Predicts the CoKG model response over the generated points.

**Returns:**

- Tuple containing predicted response and standard deviation.

#### `plot_models(self, fHF, fLF, Xe, Xc, f_pred)`

Plots the LF, HF, and CoKG models with their sampling points.

**Parameters:**

- `fHF`, `fLF`: Responses from HF and LF models.
- `Xe`, `Xc`: Sampling points for HF and LF models.
- `f_pred`: CoKG predictions.

---

### Usage Example

```python
MFA = MultiFidelityAnalysis()
Xe = np.array([[0.2], [0.4], [0.85]])
Xc = np.vstack((np.array([[0.1], [0.25], [0.3], [0.5], [0.6], [0.7], [0.8], [0.9]]), Xe))
MFA.fit_coKG(Xe, Xc)
f_pred = MFA.predict_coKG()
MFA.plot_models(fHF, fLF, Xe, Xc, f_pred)
```

---

For further inquiries or contributions, feel free to reach out.

### Author

M. Giselle Fernández-Giselle, fernandez48@llnl.gov

### License

This project is licensed under the MIT License.

---

This example initializes a `MultiFidelityModel` with 200 LFM samples and 20 HFM samples. Subsequently, it generates data, fits the SVR and RF models, and plots the outcomes.
# Multi-Fidelity Modeling Toy Problem 5: Multi-Fidelity Forrester Function 

This Python class encapsulates the functionalities related to the multi-fidelity Forrester function. It provides methods for function evaluation, plotting, and model training.

## Class Definition

### `MultiFidelityForrester`

A class for encapsulating the functionalities related to the multi-fidelity Forrester function.

#### Methods

##### `__init__(self, lb=0, ub=1, num_points=100)`

Initializes the class with default lower and upper bounds and the number of points.

- **Parameters:**
  - `lb` (float): Lower bound for function evaluation
  - `ub` (float): Upper bound for function evaluation
  - `num_points` (int): Number of linearly spaced points

##### `func_HF(self, x)`

Calculates the high-fidelity function \( f_{\text{HF}}(x) \).

- **Parameters:**
  - `x` (float): Input value
- **Returns:**
  - float: \( f_{\text{HF}}(x) \) value

##### `func_LF(self, x)`

Calculates the low-fidelity function \( f_{\text{LF}}(x) \).

- **Parameters:**
  - `x` (float): Input value
- **Returns:**
  - float: \( f_{\text{LF}}(x) \) value

##### `plot_functions(self)`

Plots both the high-fidelity and low-fidelity functions. Saves the plot as a high-resolution PNG file.

---

## Example Usage

```python
# Instantiate the class and plot the functions
if __name__ == '__main__':
    mf_forrester = MultiFidelityForrester()
    mf_forrester.plot_functions()
```

---

This documentation is intended to provide a clear and concise understanding of the `MultiFidelityForrester` class, its attributes, and its methods. The script also contains a test condition to instantiate the class and visualize the functions if it is the main program being run.

For a more extensive understanding of the algorithms used, refer to the in-line comments within the class methods.

---

Certainly. Below is the Markdown-based documentation of the Python classes you provided, each detailing their purpose, methods, and example usage.

---

# Branin Function Documentation

## BraninFunctionPlot Class

### Overview

The `BraninFunctionPlot` class serves the purpose of plotting High-Fidelity (HF) and Low-Fidelity (LF) Branin functions.

### Methods

#### `__init__(self)`

Initializes Branin function parameters.

#### `func_HF(self, x, y)`

Computes the HF function based on the Branin function.

- **Parameters:**
  - `x` (float): The x-coordinate
  - `y` (float): The y-coordinate
- **Returns:**
  - float: The value of HF function

#### `func_LF(self, x, y)`

Computes the LF function derived from the HF function.

- **Parameters:**
  - `x` (float): The x-coordinate
  - `y` (float): The y-coordinate
- **Returns:**
  - float: The value of LF function

#### `plot_functions(self)`

Generates and saves plots for both HF and LF functions.

---

## BraninFunctionModeling Class

### Overview

The `BraninFunctionModeling` class focuses on fitting additive and multiplicative models to HF and LF functions.

### Methods

#### `__init__(self)`

Initializes the polynomial features and linear regression models.

#### `func_HF(self, x, y)`

Computes the HF function based on the Branin function.

#### `func_LF(self, x, y)`

Computes the LF function derived from the HF function.

#### `fit_models(self, X_sampHF)`

Fits additive and multiplicative models based on sampled points from the HF function.

- **Parameters:**
  - `X_sampHF` (array): The sampled points from the HF function.

#### `plot_models(self)`

Generates plots for the additive and multiplicative models along with their Mean Absolute Percentage Errors (MAPE).

---

## BraninScatterPlot Class

### Overview

The `BraninScatterPlot` class plots scatter plots of HF and LF functions.

### Methods

#### `__init__(self, x_sampHF, y_sampHF)`

Initializes with the sampled points for the HF model.

- **Parameters:**
  - `x_sampHF` (array): The x-coordinates for the sampled points
  - `y_sampHF` (array): The y-coordinates for the sampled points

#### `func_HF(self, x, y)`

Computes the HF function based on the Branin function.

#### `func_LF(self, x, y)`

Computes the LF function derived from the HF function.

#### `plot_scatter(self)`

Generates scatter plots for LF and HF models with varying sizes based on the HF/LF ratio.

---

### Example Usage

```python
# For BraninFunctionPlot
branin_plotter = BraninFunctionPlot()
branin_plotter.plot_functions()

# For BraninFunctionModeling
model = BraninFunctionModeling()
model.fit_models(X_sampHF)
model.plot_models()

# For BraninScatterPlot
plotter = BraninScatterPlot(x_sampHF, y_sampHF)
plotter.plot_scatter()
```

---

This documentation aims to offer a precise yet comprehensive understanding of the functionalities each class provides. For more nuanced details, consult the in-line comments within the class methods.
