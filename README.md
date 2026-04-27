# varcomptest-jl

Julia package for fitting variance components models and testing linear hypotheses about variance components. 
This package currently supports reproduction of the results in the corresponding methods paper which is pending
submission. This README will be updated with the link to the preprint when it's available.

The package uses the formula interface of the `MixedModels` `julia` [package](https://juliastats.org/MixedModels.jl/dev/).
For example, to fit a nested split-plot model to the famous `Oats` data, you would do:

```
using 
  MixedModels,
  CategoricalArrays,
  varcomptest
# Load the data
oats = dataset("MASS", "oats");
# Format the variables into datatypes required for the @formula construct from MixedModels
oats.B = categorical(oats.B, ordered = false, levels = unique(oats.B));
oats.N = categorical(oats.N, ordered = false, levels = unique(oats.N));
oats.V = categorical(oats.V, ordered = false, levels = unique(oats.V));
oats.Y = Float64.(oats.Y);
oats.BinV = string.(oats.B) .* ":" .* string.(oats.V)
oats.BinV = categorical(oats.BinV, ordered = false, levels = unique(oats.BinV));
# Make the mixed model formula
ff = @formula(Y ~ N + V + (1 | B) + (1 | BinV))
# Set up the varcomptest. This performs 1000 bootstrap samples to test the hypothesis
control = VarCompControl(NewtonControl(verbose = false), 1000)
# Test the hypothesis that Atau = 0 where tau is the vector of variance components from the above formula.
# So this tests H0: tau1 = tau2
A = Matrix([1., -1.]')
# Fit the model
vmod = varcompmodel(ff, oats, control = control, A = A)
# Inspect the results with e.g. propertynames(vmod).
# Everything has a print method and should display nicely in the terminal.
```
