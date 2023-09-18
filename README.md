# ReviewOfMultiFidelityModels_ToyProblems
This repository comprises Jupyter Notebooks that serve as supplementary material to the journal article titled "Review of Multifidelity Models." The notebooks contain Python-based implementations that demonstrate toy problems in the multifidelity domain. 



# Multi-Fidelity Modeling Toy Problem 4: Multi-Fidelity Forrester Function 

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