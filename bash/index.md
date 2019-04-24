
  * [ ] TO DO: Digest Bash Hacker's Wiki
    * [brackets](https://dev.to/rpalo/bash-brackets-quick-reference-4eh6)
    * [ ] [getopts_tutorial](http://wiki.bash-hackers.org/howto/getopts_tutorial)
    * [ ] [brace](http://wiki.bash-hackers.org/syntax/expansion/brace)
    * [ ] [tilde](http://wiki.bash-hackers.org/syntax/expansion/tilde)
    * [ ] [conditional_expression](http://wiki.bash-hackers.org/syntax/ccmd/conditional_expression)
    * [ ] [dissectabadoneliner](http://wiki.bash-hackers.org/howto/dissectabadoneliner)
    * [ ] [calculate-dc](http://wiki.bash-hackers.org/howto/calculate-dc)
    * [ ] [redirection_tutorial](http://wiki.bash-hackers.org/howto/redirection_tutorial)
    * [ ] [collapsing_functions i.e. JavaScript-tier goodness](http://wiki.bash-hackers.org/howto/collapsing_functions)
    * [ ] [edit-ed](http://wiki.bash-hackers.org/howto/edit-ed)
    * [ ] [conffile](http://wiki.bash-hackers.org/howto/conffile)
    * [ ] [mutex](http://wiki.bash-hackers.org/howto/mutex)
    * [ ] [terminalcodes](http://wiki.bash-hackers.org/scripting/terminalcodes)
    * [ ] [style](http://wiki.bash-hackers.org/scripting/style)

# Bash Shell Scripts
Bash (**B**ourne **A**gain **Sh**ell) is one of the most widely available and used command line shell applications. Along with basic shell functionality, it offers a wide variety of features which, if utilized thoughtfully, can create powerful automated execution sequences that run software, manipulate text and files, parallelize otherwise single-process software, or anything else you may want to do from the command line. This article lists various syntaxes and common script modules to serve as a high-level resource for creating effective and succinct scripts that behave as you intend. This page also catalogs some of the more obscure features that Bash offers and attempts to provide example situations where they may be of use.

By a large margin, shell scripts are the most common way our HPC community submits jobs. Usually running a large parallel workload requires some initialization of the software environment before revving up the CPUs. This usually involves declaring environment  variables(  link to env variables), creating files that the software will run on, loading modules and libraries that the software needs to run, etc. Bash can even be used to launch several single-core jobs, effectively taking on the roll of an ad hoc batch executer. Below is (in no particular order) a list of tips, tricks, good practices, and common pitfalls when it comes to writing effective Bash scripts.

  

* * *

## For Beginners and the Unfamiliar

## Gotchas

`export`, `source`, and `declare`

* `if true;`
* `if [ true ];`
* `if [ "true" ];`
* `if false;`
* `if [ false ];`
* `if [ "false" ];`

`` `echo hi` ``, `$(echo hi)`, and `$((echo hi))`

`[ cmd ]`, `[[ cmd ]]` and `(( cmd ))`

### How Bash Works

If you read a bash script, you may default to your usual (or nonexistent) understanding of how code generally words--that is the binary/kernel which digests the code you write (compilers for C, cython interpreter for python, Java Virtual Machine for Java, etc.) interprets the text into some sort of data structure which enforces the priority of certain commands over others (much like PEMDOS for math) and generates some execution of operations based on that data structure. Bash is not quite as fancy, as many aspects of its "language" are actually just the names of compiled binaries which do the heavy lifting. Much the same way you can run `python` or `ssh` in a command line, under the hood normal bash operations such as `if`, `echo`, and `exit` are actually just programs that expect a certain cadence for the arguments you give it. A block such as:

`if [[ true ]]; then echo "If this wasn't echoed, something has gone horribly awry."`

Is really just a sequence of executing many compiled applications with arguments, the names of these commands were just chosen to give the appearance of a proper scripting language. A good example are the programs `[[`and `]]` which are indeed actual, compiled binaries written in C. This is why you need to have a space between `[[` and `]]` and your conditional, whereas in languages like C it's common to write the syntax as `if (conditional) { ...; }`. Otherwise, if you try to run `if [[TRUE]]` you will likely get an error telling you that there isn't a program called `[[TRUE]]` that you can run. This is also why you often see stray semicolons that seem somewhat arbitrary, as semicolons separate the execution of two binaries. Take this snippet for example:

`echo "First message." ; echo "Second message."`

This is identical to:

`echo "First message."`

`echo "Second message."`

In the first snippet, if the semicolon was not present, the second `echo` would be interpreted as an argument to the first echo and would end up printing: `First message. echo Second message.`

Similarly, normal if-then-else control flow that you would expect of any programming/scripting language has the same caveats. Consider this snippet:

  

  

if \[\[ true \]\]

then

echo "true is true"

else

echo "false is true?"

fi

  

If we break down what's essentially happening here (omitting some of the technical details) what's happening is:

*   *   `if` invokes `[[` (which is an alias for the program `test`) which takes `true` as logical expression to evaluate. `]]` indicates the end of the logical expression.
    *   `then` will execute anything it's given until another `if`, `else`, `elif`, or `fi` if `[[` returns a normal exit code.
    *   `else` will execute if `[[` returned an erroneous exit code.
    *   `fi` indicates that no more conditional branches exist relative to the logical expression given to the original `if`.

All this to say, this is why you often see if-then-else blocks written succinctly as `if [[ CONDITIONAL ]]; then stuff; fi` with seemingly arbitrary semicolons. It is exactly why things work this way that bash is able to execute arbitrary executables (some of which _you_ may end up writing) and not require something like Python's subprocess module.

This is just to give you an understanding for _why_ some of the syntax you will encounter is the way it is.

  

Executing/Invoking Scripts
--------------------------

What's the point of writing a script if you can't run it? First, your script needs to exist as a file on the system.

`#!/bin/bash`

  

  

  

Related articles

Related articles appear here based on the labels you select. Click to edit the macro and add or change labels.

false 5 HPCWIKI false modified true page label in ("scripting-tools","howto","scripts","shell","script","how-to","bash") and type = "page" and space = "HPCWIKI" scripts scripting-tools shell bash script how-to howto

  

true

  

Related issues