---
title: "Intro to Bash Scripting"
categories: tutorial
layout: default
parent: General
---

# An Introduction to Bash Scripting


Bash (**B**ourne **A**gain **Sh**ell) is one of the most widely available and used command line shell applications. Along with basic shell functionality, it offers a wide variety of features which, if utilized thoughtfully, can create powerful automated execution sequences that run software, manipulate text and files, parallelize otherwise single-process software, or anything else you may want to do from the command line. 

Shell scripts are also one of the most common ways our HPC community submits jobs, and running a large parallel workload often requires some initialization of the software environment before meaningful computations can begin. This typically involves tasks such as declaring environment variables, preparing input files or staging directories for data, loading modules and libraries that the software needs to run, preparing inputs, manipulating datasets, and so on. Bash can even be used to launch several single-core jobs, effectively taking on the role of an ad hoc batch executor, as well. 

This article provides a brief introduction to bash, as well as a list of tips, tricks, and good practices when it comes to writing effective bash scripts that can apply widely in both HPC and non-HPC environments. We will also provide links to some additional resources to help further your bash scripting skills.


# Executing/Invoking Scripts
All of bash commands work at the command prompt "live", i.e. interpreted line-by-line as you type commands and press enter. A bash "script" may be regarded as a list of bash commands that have been saved to a file for convenience, usually with some basic formatting, and possibly comments, for legibility.

All bash scripts must begin with a special character combination, called the "shebang" or `#!` character, followed by the name of an interpreter:

`#!/bin/bash`

This declares that the contents of the file that follow are to be interpreted as commands, using `/bin/bash` as the interpreter. This includes commands, control structures, and comments.

Plenty of other interpreters exist. For example, Python scripts begin with: `#!/usr/bin/python` or `/usr/bin/env python`, perl scripts: `#!/usr/bin/perl`, and so on.

# Bash Scripting Syntax

If you read a bash script, you may be tempted to default to your usual understanding of how code generally works. For example, with most languages, typically there is a binary or kernel which digests the code you write (compilers/gcc for C, the python interpreter/shell, Java Virtual Machine for Java, and so on.) The binary/kernel/interpreter then interprets the text into some sort of data structure which enforces the priority of certain commands over others, and finally generates some execution of operations based on that data structure.  

Bash isn't too far off from this model, and in some respects functions as any other interpreted language: you enter a command (or a control structure) and it is executed. 

However, as a shell that also serves as your major interface to the underlying operating system, it does have some properties and features that may blur the lines between what you think of as 'interpreted' versus 'compiled'.

For instance, many aspects of the bash "language" are actually just the names of pre-compiled binaries which do the heavy lifting. Much the same way you can run `python` or `ssh` in a command line, under the hood normal bash operations such as `if`, `echo`, and `exit` are actually just programs that expect a certain cadence for the arguments you give it. A block such as:

```bash
if true; then echo "true was true"; fi
```
This is really just a sequence of executing many compiled applications or shell built-ins with arguments; the names of these commands were just chosen to read as a typical programming grammar. 

A good example is the program `[` which is just an oddly-named command you can invoke. Try running `which [` at a command prompt. The results may surprise you: `/usr/bin/[` is actually a compiled program on disk, not a "built-in" function!

This is why you need to have a space between the brackets and your conditional, because the conditional itself is passed as an argument to the command `[`. In languages like C it's common to write the syntax as `if (conditional) { ...; }`. However, in bash, if you try to run `if [true]` you will likely get an error saying there isn't a command called `[true]` that you can run. This is also why you often see stray semicolons that seem somewhat arbitrary, as semicolons separate the execution of two binaries. Take this snippet for example:
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

* `if` invokes the command `true` which always exits with a successful exit code (`0`)
* `if` interprets a success exit code (`0`) as a truism and runs the `then`.
* the `then` command will execute anything it's given until `else`, `elif`, or `fi`
* the `else` command is the same as `then` but will only execute if `if` returned an erroneous exit code.
* the `fi` command indicates that no more conditional branches exist relative to the logical expression given to the original `if`.

All this to say, this is why you often see if-then-else blocks written succinctly as `if [ <CONDITIONAL> ]; then <COMMANDS>; fi` with seemingly arbitrary semicolons and spaces. It is exactly why things work this way that bash is able to execute arbitrary executables (some of which _you_ may end up writing) and not require something like Python's subprocess module.

This is just to give you an understanding for _why_ some of the syntax you will encounter is the way it is. Everything in bash is either a command or an argument to a command.


# Parentheses, Braces, and Brackets

Bash utilizes many flavors of symbolic enclosures. A complete guide is beyond the scope of this document, but you may see the following:

* `( )` - Single parentheses: run enclosed commands in a subshell
	* `a='bad';(a='good'; mkdir $a); echo $a` 
	result: directory "good" is made, echoes "bad" to screen
* `$( )` - Single parentheses with dollar sign: subshell output to string(command substitution) (preferred method)
	* `echo "my name is $( whoami )"`
	result: prints your username
* `<( )` - Parentheses with angle bracket: process substitution
	* `sort -n -k 5 <( ls -l ./dir1) <(ls -l ./dir2)`
	result: sorts ls -l results of two directories by column 5 (size)
* `[ ]` - Single Brackets: truth testing with filename expansion or word splitting
	* `if [ -e myfile.txt ]; then echo "yay"; else echo "boo"; fi`
	result: if myfile.txt exists, celebrate
* `{ }` - Single Braces/curly brackets: expansion of a range
* `${ }` - Single braces with dollar sign: expansion with interpolation
* `` ` ` `` - Backticks: command/process substitution
* `(( ))` - Double parentheses: integer arithmetic 
* `$(( ))` - Double parentheses with dollar sign: integer arithmatic to string
* `[[ ]]` - Double brackets: truth testing with regex 
  
### Additional Notes on `( )` (Single Parentheses)

There are 3 features in Bash which are denoted by a pair of parentheses, which are Bash subshells, Bash array declarations, and Bash function declarations. See the table below for when each feature is enacted:

| Syntax | Bash Feature |
|-------------------------------------------------	| ------------------------------------------------------------------------------------------------------	|
| Command/line begins with `(` | Run the contained expression(s) in a subshell. This will pass everything until a closing `)` to a child-fork of Bash that inherits the environment from the invoking Bash instance, and exits with the exit code of the last command the subshell exitted with. See the section on [subshells](#subshells) for more info. |
| A valid Bash identifier is set equal to a parnethetically enclosed list of items<br>(.e.g. `arr=("a" "b" "c")` )                             	| Creates a Bash array with elements enclosed by the parentheses. The default indexing of the elements is numerically incremental from 0 in the given order, but this order can be overridden or string-based keys can be used. See the section on [arrays](#arrays) for more info. |
| A valid Bash identifier is followed by `()` and contains some function(s) enclosed by `{ }`<br>(i.e. `func() { echo "test"; } ` ) | Declare a function which can be re/used throughout a Bash script. See the either of ["`{ }`"](#--single-braces) or [functions](#functions) for more info. | 


## Examples of Enclosure Usage

Note that whitespace is required, prohibited, or ignored in certain situations. See this block for specific examples of how to use whitespace in the various contexts of parantheses.
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

| Command | Behavior |
|-------------------------------------------------	|------------------------------------------------------------------------------------------------------	|
| `(ls -1 | head -n 1)` | Run the command in a subshell. This will return the exit code of the last process that was ran. |
| `test_var=(ls -1)` | Create a bash array with the elements `ls` and `-1`, meaning `${test_var[1]}` will evaluate to `-1`. 	|
| `test_var=$(ls -1)` | Evaluate `ls -1` and capture the output as a string. |
| ``test_var=(`ls -1`)`` or `test_var=($(ls -1))` 	| Evaluate `ls -1` and capture the output as an array. |


### Bracket Usage:
Correct:

* `[ cmd ]` - There must be spaces or terminating characters (`\n` or `;`) surrounding any brackets. 
 
* Like many common bash commands, "[" is actually a standalone executable, usually located at `/usr/bin/[`, so it requires spaces to invoke correctly. 

Erroneous:

* `[cmd]`   - tries to find a command called `[cmd]` which likely doesn't exist
* `[cmd ]`  - tries to find a command called `[cmd` and pass `]` as an argument to it
* `[ cmd]`  - tries to pass `cmd]` as an argument to `[` which expects an argument of `]` that isn't technically provided.

There are many other examples of using enclosures in bash scripting beyond the scope of this introduction. Please see the resources section for more information.

## Variables

Variable assignment in bash is simply to assign a value to a string of characters. All subsequent references to that variable must be prefixed by `$`:

```bash
$ MYSTRING="a string"
$ echo $MYSTRING
a string
$ MYNUMBER="42"
$ echo $MYNUMBER
42
```

### Exporting Variables 
When you declare a variable in bash, that variable is only available in the shell in which it is declared; if you spawn a sub-shell, the variable will not be accessible. Using the `export` command, you can essentially declare the variable to be inheritable.

```bash
# without exporting:
$ TESTVAR=100  
$ echo $TESTVAR
100     # returns a result
$ bash  # spawn a sub-shell
$ echo $TESTVAR
        # no result
$ exit  # exit the subshell
# with exporting: 
$ export TESTVAR=100
$ echo $TESTVAR
100     # returns a result 
$ bash  # spawn a sub-shell
$ echo $TESTVAR  
100     # value is passed into the subshell
$ exit  # exit the subshell
$
```
 
### Sourcing Variables
"Source" (shortcut: `.`) is a built-in bash command that takes a bash script as an argument. Bash will execute the contents of that file in the _current_ shell, instead of spawning a sub-shell. This will load any variables, function declarations, and so on into your current shell. 

A common example of using the `source` command is when making changes to your `~/.bashrc`, which is usually only parsed once upon login. Rather than logging out and logging back in every time you wish to make a change, you can simply run `source ~/.bashrc` or `. ~/.bashrc` and the changes will take effect immediately.

### Declaring Variables
Variable typing in bash is implicit, and the need to declare a type is rare, but the `declare` command can be used when necessary:
```bash
$ declare -i MYNUMBER # set type as an integer
$ echo $MYNUMBER
0
$ declare -l MYWORD="LOWERCASE" # set type as lowercase 
$ echo $MYWORD
lowercase
$
```
see `help declare` at the command line for more information on types that can be declared.


Further Resources
--------------------------

[NREL HPC Github](https://github.com/NREL/HPC/tree/code-examples/general/beginner/bash) - User-contributed bash script and examples that you can use on HPC systems.

[BASH cheat sheet](https://github.com/NREL/HPC/blob/code-examples/general/beginner/bash/cheatsheet.sh) - A concise and extensive list of example commands, built-ins, control structures, and other useful bash scripting material.




