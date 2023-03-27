---
title: Tour of Julia
postdate: November 9, 2022
layout: default
author: Jonathan Maack
description: A brief introduction to Julia
parent: Julia
grand_parent: Languages
---

# Tour of Julia

*"Julia aims to create an unprecedented combination of ease-of-use, power, and efficiency in a single language." --Julia Documentation*

## Why Julia?

Feature Highlights:

* Designed for scientific computing
* Non-vectorized code is just as fast as vectorized code
* Designed for distributed and parallel computing
* Call C/FORTRAN functions directly
* Metaprogramming

## Basics

* [REPL (Read-Evaluate-Print-Loop)](#repl-read-evaluate-print-loop)
* [Definining Functions](#definining-functions)
* [Using Installed Packages](#using-installed-packages)
* [Vectorizing](#vectorizing)

### REPL (Read-Evaluate-Print-Loop)

* Command line julia interface
* Type the command `julia` in a terminal (assuming Julia is in your path)
* Basic way to interact with objects, packages and environments

```julia
jmaack-32918s:~ jmaack$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.6.1 (2021-04-23)
 _/ |\__'_|_|_|\__'_|  |  
|__/                   |
```
```julia
julia> 4 * pi^2 + sqrt(2)im
39.47841760435743 + 1.4142135623730951im

help?> Int
search: Int Int8 Int64 Int32 Int16 Int128 Integer intersect intersect! InteractiveUtils InterruptException

  Int64 <: Signed

  64-bit signed integer type.

julia> exit()
```

**NOTE:** When using the REPL, the result of the (last) expression is always printed. This is sometimes undesirable. We can supress printing by ending the last expression with a semicolon `;`. This is used throughout this presentation for appearance purposes. *Unless otherwise stated any semicolon in code is not needed.*

### Defining Functions

There are two ways to define functions

* Standard way:

    ```julia
    function my_function(x)
        return x^2
    end;
    ```

* Short form way:

    ```julia
    my_func(x) = x^2;
    ```

It is also possible to define anonymous functions (and save pointers to them):


```julia
f = (x)->x^2;
```


```julia
@show my_function(pi)
@show my_func(pi)
@show f(pi);
```

    my_function(pi) = 9.869604401089358
    my_func(pi) = 9.869604401089358
    f(pi) = 9.869604401089358


**NOTE:** Julia uses the standard control flow keywords such as `for`, `while`, `if`, `elseif`, `else`. See the [Control Flow](https://docs.julialang.org/en/v1/manual/control-flow/) section of the Julia documentation for more details. Obviously, these are helpful in writing functions.

### Using Installed Packages

Packages can be accessed in two ways:

* `import` statement -- makes all module attributes (i.e. functions and types) available by prefixing the module name followed by a dot

    ```julia
    x = rand(5)
    import Statistics
    Statistics.mean(x)
    ```




        0.3339056277968421



* `using` statement -- everything exported by the module is directly accessible

    ```julia
    using Statistics
    mean(x)
    ```




        0.3339056277968421




Any attribute that is not exported by the module can still be accessed by prefixing the module name followed by a dot.


```julia
Statistics._conj(x)
```




    5-element Vector{Float64}:
     0.17922586649673145
     0.7155842248637634
     0.29280412953665125
     0.10325841440419592
     0.3786555036828685



**NOTE:** Like in python, there are no private attributes. Users may access anything created by a module. Package authors can suggest attributes that users should not use by not exporting them or with naming conventions (e.g. prefixing `_` to any name that is internal only).

Julia 1.6 introduced the "pythonic" import syntax

```julia
import Statistics as Stats
Stats.mean(x)
```




    0.3339056277968421



In older Julia versions, we can declare a constant for our packages

```julia
import Statistics
const St = Statistics
St.mean(x)
```




    0.3339056277968421



**ADVICE:** When writing Julia code, use `import` rather than `using`. This makes code easier to follow as well as giving hints on where to look for documentation.

### Vectorizing

Julia uses the MATLAB dot syntax to operate component-wise on arrays (i.e. vectors and matrices)


```julia
x = rand(3)
y = rand(3)
(x.*y).^2
```




    3-element Vector{Float64}:
     0.5367929263482071
     0.008092183589557244
     0.36146876615689527



Julia also extends this syntax to **ANY** function that operates on vector elements


```julia
number_op(x) = x + 5
number_op.(x)
```




    3-element Vector{Float64}:
     5.754141942494573
     5.8412967567631
     5.637813968303307




In Julia, vectorizing is done for convenience rather than performance:


```julia
function my_mult_for(x,y)
    z = zeros(length(x))
    for k in length(x)
        z[k] = x[k] * y[k]
    end
    return z
end

function my_mult_vect(x,y)
    return x .* y
end;
```


```julia
# This forces Julia to compile the function definitions
# so that the timing results in the next cell are correct
x = rand(2)
y = rand(2)
@time my_mult_vect(x,y)
@time my_mult_for(x,y);
```

      0.055219 seconds (145.07 k allocations: 8.243 MiB, 99.96% compilation time)
      0.009099 seconds (15.42 k allocations: 873.090 KiB, 99.82% compilation time)



```julia
x = rand(10000)
y = rand(10000)
@time my_mult_vect(x,y)
@time my_mult_for(x,y);
```

      0.000015 seconds (2 allocations: 78.203 KiB)
      0.000032 seconds (2 allocations: 78.203 KiB)


## Package Manager

* [Managing Packages (REPL)](Managing-Packages-(REPL))
* [Managing Packages (Scripts)](Managing-Packages-(Scripts))
* [Environments](Environments)
* [Activating Environments](Activating-Environments)
* [Copying Environments](Copying-Environments)
* [Environment Layering](Environment-Layering)

### Managing Packages (REPL)

Open the REPL and hit the `[` key to enter package management mode. From here we can add or remove packages:

```julia
(@v1.6) pkg> add Compat
   Resolving package versions...
    Updating `~/.julia/environments/v1.6/Project.toml`
  [34da2185] + Compat v3.31.0
    Updating `~/.julia/environments/v1.6/Manifest.toml`
  [34da2185] + Compat v3.31.0
  [8bb1440f] + DelimitedFiles
  [8ba89e20] + Distributed
  [1a1011a3] + SharedArrays
  [2f01184e] + SparseArrays
  [10745b16] + Statistics

(@v1.6) pkg> rm Compat
    Updating `~/.julia/environments/v1.6/Project.toml`
  [34da2185] - Compat v3.31.0
    Updating `~/.julia/environments/v1.6/Manifest.toml`
  [34da2185] - Compat v3.31.0
  [8bb1440f] - DelimitedFiles
  [8ba89e20] - Distributed
  [1a1011a3] - SharedArrays
  [2f01184e] - SparseArrays
  [10745b16] - Statistics
```


We can also print out what packages are available
```julia
(@v1.6) pkg> st
      Status `~/.julia/environments/v1.6/Project.toml`
  [7073ff75] IJulia v1.23.2
  [438e738f] PyCall v1.92.3
```
or update the packages
```julia
(@v1.6) pkg> up
    Updating registry at `~/.julia/registries/General`
    Updating git-repo `https://github.com/JuliaRegistries/General.git`
  No Changes to `~/.julia/environments/v1.6/Project.toml`
  No Changes to `~/.julia/environments/v1.6/Manifest.toml`
```

### Managing Packages (Scripts)

Package management mode in the REPL is actually just a convenient interface to the Julia package [Pkg.jl](https://pkgdocs.julialang.org/stable/) which is part of the Julia standard library.

All package mode commands are functions in Pkg.jl:

```julia
import Pkg; Pkg.add("Compat"); Pkg.rm("Compat")

    Updating registry at `~/.julia/registries/General`
    Updating git-repo `https://github.com/JuliaRegistries/General.git`
   Resolving package versions...
    Updating `~/.julia/environments/v1.6/Project.toml`
  [34da2185] + Compat v3.31.0
    Updating `~/.julia/environments/v1.6/Manifest.toml`
  [34da2185] + Compat v3.31.0
  [8bb1440f] + DelimitedFiles
  [8ba89e20] + Distributed
  [1a1011a3] + SharedArrays
  [2f01184e] + SparseArrays
  [10745b16] + Statistics
    Updating `~/.julia/environments/v1.6/Project.toml`
  [34da2185] - Compat v3.31.0
    Updating `~/.julia/environments/v1.6/Manifest.toml`
  [34da2185] - Compat v3.31.0
  [8bb1440f] - DelimitedFiles
  [8ba89e20] - Distributed
  [1a1011a3] - SharedArrays
  [2f01184e] - SparseArrays
  [10745b16] - Statistics
```

```julia
Pkg.status(); Pkg.update()

      Status `~/.julia/environments/v1.6/Project.toml`
  [7073ff75] IJulia v1.23.2
  [438e738f] PyCall v1.92.3
    Updating registry at `~/.julia/registries/General`
    Updating git-repo `https://github.com/JuliaRegistries/General.git`
  No Changes to `~/.julia/environments/v1.6/Project.toml`
  No Changes to `~/.julia/environments/v1.6/Manifest.toml`
```

**WARNING:** If you want to use Julia within Jupyter notebook, some package management features (like adding new packages) do not work well. It is best to add/remove/update either with a script or using the REPL.

### Environments

Environments allow us to install different versions of packages for use with different projects. Very similar to python virtual environments or conda environments.

```julia
Pkg.activate("env-one"); Pkg.status()

  Activating environment at `~/HPC_Apps/julia-tutorial/env-one/Project.toml`
      Status `~/HPC_Apps/julia-tutorial/env-one/Project.toml`
  [91a5bcdd] Plots v1.13.1
```

```julia
Pkg.activate("env-two"); Pkg.status()

  Activating environment at `~/HPC_Apps/julia-tutorial/env-two/Project.toml`
      Status `~/HPC_Apps/julia-tutorial/env-two/Project.toml`
  [91a5bcdd] Plots v1.16.6
```

The environment names are given by the directory in which they reside. The explicitly added packages are given in the `Project.toml` file. The entire environment with all the required dependencies (down to specific commits) are in the `Manifest.toml` file.

### Activating Environments

There are 3 ways to activate an environment:

* Using the `Pkg.activate` function:
    ```julia
    Pkg.activate("path/to/environment/")
    ```
* Within package management mode with the `activate` command:
    ```julia
    activate path/to/environment
    ```
* From the command line with the `--project` option:
    ```shell
    julia --project=<path/to/environment>
    ```

The first 2 ways can also be used to **create** new environments.

### Copying Environments

To copy an environment, all you need is the `Project.toml` file. Put it in the desired directory and activate that environment. Finally, in package management mode, use the `instantiate` command:

```julia
(fake-env) pkg> st
      Status `~/fake-env/Project.toml`
→ [da04e1cc] MPI v0.18.1
        Info packages marked with → not downloaded, use `instantiate` to download

(fake-env) pkg> instantiate
   Installed MPI ─ v0.18.1
    Building MPI → `~/.julia/scratchspaces/44cfe95a-1eb2-52ea-b672-e2afdf69b78f/494d99052881a83f36f5ef08b23de07cc7c03a96/build.log`
Precompiling project...
  1 dependency successfully precompiled in 2 seconds (11 already precompiled)
```

**NOTE:** Alternatively, you can use the `Pkg.instantiate` function.

**NOTE:** If you need to exactly copy an environment exactly copy both the `Project.toml` and `Manifest.toml` files into the desired directory and use the `instantiate` command.

### Environment Layering

Julia environments can be layered such that packages from more than just the top layer environment can be imported. This allows us to have access to debugging and development tools without putting them in whatever environment were working on. This is a major difference from conda environments.

```julia
Pkg.status()
      Status `~/HPC_Apps/julia-tutorial/env-one/Project.toml`
  [91a5bcdd] Plots v1.13.1
```


```julia
import BenchmarkTools as BT # THIS IS NOT IN OUR TOP ENVIRONMENT!!!
```

When loading a package, Julia has a hierarchy of environments that it checks for the package. Julia loads the first version of the package it encounters in this hierarchy.  The environment hierarchy can be altered by the [`JULIA_LOAD_PATH`](https://docs.julialang.org/en/v1/manual/environment-variables/) environment variable.

These environment stacks are discussed more in the [Environments](https://docs.julialang.org/en/v1/manual/code-loading/#Environments) subsection of the Code Loading part of the Julia Manual.

## Types

* [Type Hierarchy](#type-hierarchy)
* [Multiple Dispatch](#multiple-dispatch)

### Type Hierarchy

In Julia everything has a type. We can access an object's type with the `typeof` function:


```julia
typeof(7.5)
```




    Float64



Even types have a type:


```julia
typeof(Float64)
```




    DataType




Julia also has a type hierarchy. There are subtypes and supertypes. We can access explore these with the functions `subtypes` and `supertype`:


```julia
subtypes(Float64)
```




    Type[]




```julia
supertype(Float64)
```




    AbstractFloat



`Float64` has no subtypes because it is a **Concrete Type**. All the supertypes are an **Abstract Type**.  Only Concrete Types can actually exist.


Every type has **only one** immediate supertype. However, each supertype has a supertype. We can get the whole chain with the `supertypes` (plural) function:


```julia
supertypes(Float64)
```




    (Float64, AbstractFloat, Real, Number, Any)



Let us see all the floating point types available in Julia:


```julia
subtypes(AbstractFloat)
```




    4-element Vector{Any}:
     BigFloat
     Float16
     Float32
     Float64



We can test whether or not a type is a subtype of something with the `<:` operator:


```julia
Float64 <: AbstractFloat
```




    true




```julia
Float64 <: Float64
```




    true




```julia
Int <: AbstractFloat
```




    false



**WARNING:** Subtypes and supertypes get complicated when dealing with containers:


```julia
Float64 <: Real
```




    true




```julia
Vector{Float64} <: Vector{Real}
```




    false




```julia
Vector{Float64} <: Vector
```




    true




We can use this to write functions:


```julia
function my_abs_sub(x)
    if typeof(x) <: Complex
        println("Complex!")
        return sqrt(x.re^2 + x.im^2)
    elseif typeof(x) <: Real
        println("Real!")
        return x < 0 ? -x : x
    else
        error("Not a number!")
    end
end
@show my_abs_sub(-5)
@show my_abs_sub(-5.0)
@show my_abs_sub(-1 + 2im);
```

    Real!
    my_abs_sub(-5) = 5
    Real!
    my_abs_sub(-5.0) = 5.0
    Complex!
    my_abs_sub(-1 + 2im) = 2.23606797749979


### Multiple Dispatch

A more Julia way of doing this is to write the typing information directly into the function definition:


```julia
function my_abs_md(x::Real)
    println("Multiple Dispatch Real!")
    return x < 0 ? -x : x
end
function my_abs_md(x::Complex)
    println("Multiple Dispatch Complex!")
    return sqrt(x.re^2 + x.im^2)
end
@show my_abs_md(-5)
@show my_abs_md(-1 + 2im);
```

    Multiple Dispatch Real!
    my_abs_md(-5) = 5
    Multiple Dispatch Complex!
    my_abs_md(-1 + 2im) = 2.23606797749979


Notice that the functions have the same name, but the correct one is executed based on the type of the argument. This is called **Multiple Dispatch**.

**ADVICE:** Add typing information for any function you are likely to use a lot. There are two reasons
1. Type information is used by the Julia compiler to make code more efficient
2. Type information is a fast and easy way to document your code and catch bugs.

## Structs

* [Defining Structs](#defining-structs)
* [Mutable Structs](#mutable-structs)
* [Parametric Types](#parametric-types)

### Defining Structs

Julia allows us to define our own (composite) types:


```julia
struct Point
    x::Float64
    y::Float64
end
p0 = Point(0, 0)
p1 = Point(1.0, 2.0)
```




    Point(1.0, 2.0)



We can define functions with this type as the argument now


```julia
function distance(p::Point, q::Point)
    return sqrt((p.x - q.x)^2 + (p.y - q.y)^2)
end
distance(p0, p1)
```




    2.23606797749979




We can build structs with other structs as components:


```julia
struct Circle
    center::Point
    radius::Float64
end

my_circle = Circle(p1, 5)
```




    Circle(Point(1.0, 2.0), 5.0)




```julia
function is_in(p::Point, c::Circle)
    return distance(p, c.center) < c.radius
end
@show is_in(p0, my_circle)
@show is_in(Point(100,0), my_circle);
```

    is_in(p0, my_circle) = true
    is_in(Point(100, 0), my_circle) = false


### Mutable Structs

What if we want to change the radius of the circle?


```julia
my_circle.radius = 10.0 # Causes an error!!
```


    setfield! immutable struct of type Circle cannot be changed

    

    Stacktrace:

     [1] setproperty!(x::Circle, f::Symbol, v::Float64)

       @ Base ./Base.jl:34

     [2] top-level scope

       @ In[34]:1

     [3] eval

       @ ./boot.jl:360 [inlined]

     [4] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

       @ Base ./loading.jl:1116


Structs are immutable (cannot be changed) by default in Julia. This allows for some optimizations behind the scenes and most of the time we do not need to change the values in a Struct.


If we need to change fields in a struct, we add the `mutable` keyword:


```julia
mutable struct MutableCircle
    center::Point
    radius::Float64
end
my_mutable_circle = MutableCircle(p1, 5.0)
@show my_mutable_circle
my_mutable_circle.radius = 10.0
@show my_mutable_circle;
```

    my_mutable_circle = MutableCircle(Point(1.0, 2.0), 5.0)
    my_mutable_circle = MutableCircle(Point(1.0, 2.0), 10.0)


### Parametric Types

Let us go back to our Point type:

```julia
struct Point
    x::Float64
    y::Float64
end
```

We locked in the types in the fields of this struct. What if we want to use a `Point` struct with a different type? Such as an `Int`. We use a **Parametric Type**.


We define a Parametric Type in the following way:


```julia
struct ParametricPoint{R <: Real}
    x::R
    y::R
end

function distance(p::ParametricPoint{<:Real},
        q::ParametricPoint{<:Real})
    return sqrt((p.x - q.x)^2 + (p.y - q.y)^2)
end;
```


```julia
p0 = ParametricPoint(1, -1)
@show typeof(p0)
p1 = ParametricPoint(2.0, 0.0)
@show typeof(p1)
@show distance(p0,p1);
```

    typeof(p0) = ParametricPoint{Int64}
    typeof(p1) = ParametricPoint{Float64}
    distance(p0, p1) = 1.4142135623730951


## Metaprogramming

* [How Julia Code is Executed](#how-julia-code-is-executed)
* [Expressions](#expressions)
* [Macros](#macros)

### How Julia Code is Executed

At a very high level, Julia code is executed in two phases:

1. Parsing a string and turning it into an expression
2. Evaluating that expression

### Expressions

Julia code is parsed and turned into expressions. These expressions are themselves Julia data structures.


```julia
expr = Meta.parse("z^2 + 1")
expr
```




    :(z ^ 2 + 1)



While the expression prints as a human readable mathematical expression, it is actually a tree:


```julia
dump(expr)
```

    Expr
      head: Symbol call
      args: Array{Any}((3,))
        1: Symbol +
        2: Expr
          head: Symbol call
          args: Array{Any}((3,))
            1: Symbol ^
            2: Symbol z
            3: Int64 2
        3: Int64 1



Since this is a data structure, we can change the expression


```julia
expr.args[1] = :-
expr.args[2].args[1] = :*
expr
```




    :(z * 2 - 1)



Then evaluate it


```julia
z = 3
@show eval(expr)
z = 2.5
@show eval(expr);
```

    eval(expr) = 5
    eval(expr) = 4.0


Note we gave `z` a value **after** we wrote the expression.

### Macros

A macro is a special function that takes expressions, symbols and literal values as arguments and returns an expression. The biggest difference between a macro and a normal function is that a macro is executed during the **parse** phase. This means that in a macro we have access to the expression!

Let's take a look at the `@assert` macro:


```julia
x = 5; y = 4;
@assert x == y
```


    AssertionError: x == y

    

    Stacktrace:

     [1] top-level scope

       @ In[42]:2

     [2] eval

       @ ./boot.jl:360 [inlined]

     [3] include_string(mapexpr::typeof(REPL.softscope), mod::Module, code::String, filename::String)

       @ Base ./loading.jl:1116


The error contains the expression that caused the error! This is not possible to do with a function because that expression is not available at runtime.


How do we write macros? More or less like we write functions but using the `macro` keyword instead of the `function` keyword:


```julia
macro fadd(name::Symbol, f::Symbol, g::Symbol, nargs::Int)
    x = [gensym() for _ in 1:nargs]
    quote
        $(esc(name))($(x...)) = $(esc(f))($(x...)) + $(esc(g))($(x...))
    end
end
```




    @fadd (macro with 1 method)



This macro takes two functions and creates an expression that for a function that computes the sum of the two. It is actually generating code!


```julia
p(x) = x^2
q(x) = (2x + 5) / x^2
@fadd(h, p, q, 1)
@show p(pi) + q(pi)
@show h(pi);
```

    p(pi) + q(pi) = 11.012830091668627
    h(pi) = 11.012830091668627


We can look at the expression that the macro generates with the macro `@macroexpand`:


```julia
@macroexpand(@fadd(h, p, q, 1))
```




    quote
        #= In[43]:4 =#
        h(var"#73###258") = begin
                #= In[43]:4 =#
                p(var"#73###258") + q(var"#73###258")
            end
    end



Ignoring all the stuff with `#` symbols we can see that the expression returned by the macro looks more or less like a function definition.


Having seen how this works let's unpack the macro definition a bit more. For context, here's the whole definition again:

```julia
macro fadd(name::Symbol, f::Symbol, g::Symbol, nargs::Int)
    x = [gensym() for _ in 1:nargs]
    quote
        $(esc(name))($(x...)) = $(esc(f))($(x...)) + $(esc(g))($(x...))
    end
end
```

We'll unpack it one line at a time.


Having seen how this works let's unpack the macro definition a bit more. For context, here's the whole definition again:

```julia
macro fadd(name::Symbol, f::Symbol, g::Symbol, nargs::Int)
    x = [gensym() for _ in 1:nargs]
    quote
        $(esc(name))($(x...)) = $(esc(f))($(x...)) + $(esc(g))($(x...))
    end
end
```

*First Line:*

```julia
macro fadd(name::Symbol, f::Symbol, g::Symbol, nargs::Int)
    ...
end
```

The macro definition looks a lot like a function definition but with `macro` instead of `function`.


*Second Line:*

```julia
    x = [gensym() for _ in 1:nargs]
```

Here we create a vector of symbols of size `nargs`. The `gensym` function generates a symbol for a variable that is guaranteed not to clash with existing variables.  These symbols will be the arguments of our new function.

*Third Line:*

```julia
    quote
        # expression here
    end
```

This is an easy way to generate an expression. The contents of this block is the expression returned by the macro.

*Fourth Line:*

```julia
        $(esc(name))($(x...)) = $(esc(f))($(x...)) + $(esc(g))($(x...))
```

This is the meat of the macro and it may seem a bit much at first.  However, each term is essentially the same. So let's just focus on the left hand side of the equality.

```julia
        $(esc(name))($(x...))
```

* The `name` variable is local to the macro. It's value is what we want to put into the expression. So we **interpolate** it into the expression using `$`. 
* However, we want that symbol to be evaluated in the context in which the macro was called. So we tell Julia to leave the value as is with the `esc` function.
* Without the call to `esc`, Julia will assume that the variable is local and needs to be renamed with `gensym` transformed so that it will not clash with other variables.
* Finally, we want to interpolate the contents of the vector `x` into the expression. This is done with the **splat operator** `...` in conjunction with `$`.

**Why can't we just write a function to do this?** Let's try:

```julia
function fadd(name, f::Function, g::Function, nargs::Int)
    x = [gensym() for _ in 1:nargs]
    [WHAT HERE?](x...) = f(x...) + g(x...)
    return [WHAT TO RETURN?]
end
```

There are a couple problems here:

1. What do we put for the function name? We want the **value** of the argument name. If we just put `name` we would end up with a function called name.
2. What do we return? Even if we knew what to name the function, that name is only bound to the function **in our current scope**--the function `fadd`. Once we return from `fadd`, the name is no longer bound to this function.

If we do not care about creating function names, we could construct and return an anonymous function:


```julia
function fadd(f::Function, g::Function, nargs::Int)
    x = [gensym() for _ in 1:nargs]
    return (x...)->(f(x...) + g(x...))
end
h1 = fadd(p,q,1)
h1(pi)
```




    11.012830091668627



This gets us pretty close to the same functionality since we could assign the function pointer to any valid variable name.

However, we did not maximize the value of the macro. We can actually generate documentation for our function as well:


```julia
macro fadd(name::Symbol, f::Symbol, g::Symbol, nargs::Int)
    x = [gensym() for _ in 1:nargs]
    local help = "Functions $f and $g added together. Created with the `@fadd` macro!"
    quote
        @doc string($help)
        $(esc(name))($(x...)) = $(esc(f))($(x...)) + $(esc(g))($(x...))
    end
end
@fadd(h,p,q,1);
```


```julia
?h
```





```julia
Functions p and q added together. Created with the `@fadd` macro!
```




## Other Resources

The [Julia Documentation](https://docs.julialang.org/en/v1/) is a great place to read about Julia features. Numerous examples are normally given along with detailed explanation.

The [official Julia website](https://julialang.org/) is a great place to find [Julia tutorials](https://julialang.org/learning/), learn about the [Julia community](https://julialang.org/community/) or discover [research](https://julialang.org/research/) using Julia.
