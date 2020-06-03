# Building packages on Peregrine for individual or project use.
This training module will walk through how to build a reasonably complex package, OpenMPI, and deploy
it for use by yourself or members of a project.

1. [Acquire the package and set up for build](acquire.md)

2. [Configure, build, and install the package](config_make_install.md)

3. [Setting up your own environment module](modules.md)

## Why build your own application?
* Sometimes, the package version that you need, or the capabilities you want,
are only available as source code.
* Other times, a package has dependencies on other ones with application
programming interfaces that change rapidly. A source code build might
have code to adapt to the (older, newer) libraries you have available,
whereas a binary distribution will likely not. In other cases, a binary
distribution may be associated with a particular Linux distribution and
version different from Peregrine's. One example is a package for Linux
version X+1 (with a shiny new libc). If you try to run this on Linux
version X, you will almost certainly get errors associated with the
GLIBC version required. If you build the application against your own,
older libc version, those dependencies are not created.
* Performance; for example, if a more performant numerical library is
available, you may be able to link against it. A pre-built binary may
have been built against a more universally available but lower performance
library. The same holds for optimizing compilers.
* Curiosity to know more about the tools you use.
* Pride of building one's tools oneself.
* For the sheer thrill of building packages. :smile:

