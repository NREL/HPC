---
title: "Bash"
categories: tutorial
layout: default
parent: General
---

# Bash

* TOC
{:toc}


Bash (**B**ourne **A**gain **Sh**ell) is one of the most widely available and used command line shell applications. Along with basic shell functionality, it offers a wide variety of features which, if utilized thoughtfully, can create powerful automated execution sequences that run software, manipulate text and files, parallelize otherwise single-process software, or anything else you may want to do from the command line. This article lists various syntaxes and common script modules to serve as a high-level resource for creating effective and succinct scripts that behave as you intend. This page also catalogs some of the more obscure features that Bash offers and attempts to provide example situations where they may be of use.

Shell scripts are one of the most common ways our HPC community submits jobs. Usually running a large parallel workload requires some initialization of the software environment before revving up the CPUs. This usually involves declaring environment  variables, creating files that the software will run on, loading modules and libraries that the software needs to run, etc. Bash can even be used to launch several single-core jobs, effectively taking on the roll of an ad hoc batch executer. Below is (in no particular order) a list of tips, tricks, and good practices when it comes to writing effective Bash scripts.



# Bash Scripting Syntax

If you read a bash script, you may default to your usual (or nonexistent) understanding of how code generally words&mdash;that is the binary/kernel which digests the code you write (compilers for C, python interpreter, Java Virtual Machine for Java, etc.) interprets the text into some sort of data structure which enforces the priority of certain commands over others (much like PEMDOS for math) and generates some execution of operations based on that data structure. Bash is not quite as fancy, as many aspects of its "language" are actually just the names of compiled binaries which do the heavy lifting. Much the same way you can run `python` or `ssh` in a command line, under the hood normal bash operations such as `if`, `echo`, and `exit` are actually just programs that expect a certain cadence for the arguments you give it. A block such as:

```bash
if true; then echo "true was true"; fi
```
This is really just a sequence of executing many compiled applications or shell built-ins with arguments&mdash;the names of these commands were just chosen to read as a typical programming grammar. A good example is the program `[` which is just an oddly-named command you can invoke. This is why you need to have a space between the brackets and your conditional, because the conditional itself is passed as an argument to the command `[`. In languages like C it's common to write the syntax as `if (conditional) { ...; }`. Otherwise, if you try to run `if [true]` you will likely get an error saying there isn't a command called `[true]` that you can run. This is also why you often see stray semicolons that seem somewhat arbitrary, as semicolons separate the execution of two binaries. Take this snippet for example:
```bash
echo "First message." ; echo "Second message."
```
This is equivalent to:
```bash
echo "First message."
echo "Second message."
```
In the first snippet, if the semicolon was not present, the second `echo` would be interpreted as an argument to the first echo and would end up outputting: `First message. echo Second message.`

Bash interprets `;` and `\n` (newline) as separators. If you need to pass these characters into a function (for example, common in `find`'s `-exec` flag) you need to escape them with a `\`. This is useful for placing arguments on separate lines to improve readability like this example:
```bash
chromium-browser \
--start-fullscreen \
--new-window \
--incognito \
'https://google.com'
```

Similarly, normal if-then-else control flow that you would expect of any programming/scripting language has the same caveats. Consider this snippet:
```bash
if true
then
  echo "true is true"
else
  echo "false is true?"
fi
```
If we break down what's essentially happening here (omitting some of the technical details):
*   `if` invokes the command `true` which always exits with a successful exit code (`0`)
*   `if` interprets a success exit code (`0`) as a truism and runs the `then`.
*   the `then` command will execute anything it's given until `else`, `elif`, or `fi`
*   the `else` command is the same as `then` but will only execute if `if` returned an erroneous exit code.
*   the `fi` command indicates that no more conditional branches exist relative to the logical expression given to the original `if`.

All this to say, this is why you often see if-then-else blocks written succinctly as `if [ <CONDITIONAL> ]; then <COMMANDS>; fi` with seemingly arbitrary semicolons and spaces. It is exactly why things work this way that bash is able to execute arbitrary executables (some of which _you_ may end up writing) and not require something like Python's subprocess module.

This is just to give you an understanding for _why_ some of the syntax you will encounter is the way it is. Everything in bash is either a command or an argument to a command.


# Parentheses, Braces, and Brackets

Bash utilizes many flavors of symbolic enclosures. This section will detail the purpose, function, and nuances of what they provide.

Specifically, unique pair-wise symbols recognized by bash are:
* [`( )`](#--single-parentheses)
* [`$( )`](#--dollar-prefixed-single-parentheses)
* [`[ ]`](#--single-brackets)
* [`{ }`](#--single-braces)
* [`${ }`](#--dollar-prefixed-single-braces)
* [`` ` ` ``](#--backticks)
* [`(( ))`](#--double-parentheses)
* [`$(( ))`](#--dollar-prefixed-double-parentheses)
* [`[[ ]]`](#--double-brackets)
  
## `( )` (Single Parentheses)

There are 3 features in Bash which are denoted by a pair of parentheses, which are Bash subshells, Bash array declarations, and Bash function declarations. See the table below for when each feature is enacted:

| Syntax | Bash Feature |
|-------------------------------------------------	| ------------------------------------------------------------------------------------------------------	|
| Command/line begins with `(` | Run the contained expression(s) in a subshell. This will pass everything until a closing `)` to a child-fork of Bash that inherits the environment from the invoking Bash instance, and exits with the exit code of the last command the subshell exitted with. See the section on [subshells](#subshells) for more info. |
| A valid Bash identifier is set equal to a parnethetically enclosed list of items<br>(.e.g. `arr=("a" "b" "c")` )                             	| Creates a Bash array with elements enclosed by the parentheses. The default indexing of the elements is numerically incremental from 0 in the given order, but this order can be overridden or string-based keys can be used. See the section on [arrays](#arrays) for more info. |
| A valid Bash identifier is followed by `()` and contains some function(s) enclosed by `{ }`<br>(i.e. `func() { echo "test"; } ` ) | Declare a function which can be re/used throughout a Bash script. See the either of ["`{ }`"](#--single-braces) or [functions](#functions) for more info. | 
| | |

Note that whitespace is required, prohibited, or ignored in certain situations. See this block for specific examples of how to use whitespace in the various contexts of parantheses:
```bash
### Subshells
(echo hi)   # OK
( echo hi)  # OK
(echo hi )  # OK
( echo hi ) # OK

### Arrays
arr=("a" "b" "c")   # Array of 3 strings
arr =("a" "b" "c")    # ERROR
arr= ("a" "b" "c")    # ERROR
arr = ("a" "b" "c")   # ERROR
arr=("a""b""c")     # Array of one element that is "abc"
arr=("a","b","c")   # Array of one element that is "a,b,c"
arr=("a", "b", "c") # ${arr[0]} == "a,"

### Functions 
func(){echo hi;} # ERROR
func(){ echo hi;}     # OK
func (){ echo hi;}    # OK
func () { echo hi;}   # OK
func () { echo hi; }  # OK
```

## `$( )` (Dollar Prefixed Single Parentheses)
## `[ ]` (Single Brackets)
## `{ }` (Single Braces)
## `${ }` (Dollar Prefixed Single Braces)
## `` ` ` `` (Backticks)
## `(( ))` (Double Parentheses)
## `$(( ))` (Dollar Prefixed Double Parentheses)
## `[[ ]]` (Double Brackets)

## Nuanced Examples of Enclosure Usage


| Command | Behavior |
|-------------------------------------------------	|------------------------------------------------------------------------------------------------------	|
| `(ls -1 | head -n 1)` | Run the command in a subshell. This will return the exit code of the last process that was ran. |
| `test_var=(ls -1)` | Create a bash array with the elements `ls` and `-1`, meaning `${test_var[1]}` will evaluate to `-1`. 	|
| `test_var=$(ls -1)` | Evaluate `ls -1` and capture the output as a string. |
| ``test_var=(`ls -1`)`` or `test_var=($(ls -1))` 	| Evaluate `ls -1` and capture the output as an array. |
| | |


### Usage:
Erroneous:
* `[cmd]`   - tries to find a command called `[cmd]` which likely doesn't exist
* `[cmd ]`  - tries to find a command called `[cmd` and pass `]` as an argument to it
* `[ cmd]`  - tries to pass `cmd]` as an argument to `[` which expects an argument of `]` that isn't technically provided.

Correct:
* `[ cmd ]` - There must be spaces or terminating characters (`\n` or `;`) surrounding any brackets

#### Types



## Booleans (true/false)

* `if true;`
* `if [ true ];`
* `if [ "true" ];`
* `if false;`
* `if [ false ];`
* `if [ "false" ];`

## Variables & Arrays

`export`, `source`, and `declare`

## 

`` `echo hi` ``, `$(echo hi)`, and `$((echo hi))`



  

Executing/Invoking Scripts
--------------------------

What's the point of writing a script if you can't run it? First, your script needs to exist as a file on the system.

`#!/bin/bash`
