```@meta
EditURL = "https://github.com/simonsobs/PSpipe/tree/planckcov/project/Planck_cov/src/test.jl"
```

# Rational numbers

In julia rational numbers can be constructed with the `//` operator.
Lets define two rational numbers, `x` and `y`:

```@example test
# Define variable x and y
x = 1//3
y = 2//5
```

When adding `x` and `y` together we obtain a new rational number:

```@example test
z = x + y

# Let's plot something
using Plots
x = range(0, stop=6π, length=1000)
y1 = sin.(x)
y2 = cos.(x)

plot(x, y1)
plot!(x, y2)
```

Testing math
```math
\int f dx
```
