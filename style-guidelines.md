# Style Guidelines

### ***New to Markdown?***
Just take a look at the raw-text of [`template/README.md`](template/README.md) as a starting point for what syntaxes get rendered in what ways. Any `.md` file here can be used as a reference for how to style content.

Note that git (the version control protocol) is not equivalent to GitHub (the git repository hosting web service, which implements the git protocol). This is an important distinction to be sure of before proceeding. There are other git-hosting services such as GitLab which function similarly to GitHub, but both use the git protocol. These hosting-services provide different decorative features to make the repositories more visually pleasing/intuitive to interact with.

One such feature of GitHub's frontend is the "[Wiki](https://github.com/NREL/HPC/wiki)" (found in the tab at the top of the repository webpage). This is essentially its own child git repository that is intended to only hold template documents (like markdown files) and be more sensible to navigate as a web interface than a git repo. To edit the material in the Wiki of this repository, you will need to clone it separately (it's the same URI to clone this repository just with `.wiki` at the end.)

## Files and directories

### Naming files
* Files need an appropriate extension so it is obvious what purpose they serve. Compiled binaries shouldn't be pushed here (the most common type of extension-lacking files) so there's no reason a file should be without an extension.
  * This includes Markdown files, which should carry a ".md" extension (not .txt or lackthereof).
* No capitals or spaces (with the exception of any `README.md` files serving as landing pages within directories per GitHub markdown rendering). Separate words in the filename with underscores or hyphens&mdash;whichever reduces ambiguity the most and is most pleasant for you to type.
  * For example, using underscores in a potential filename containing a hyphenated command like `llvm-g++` would make it clear that the command has a hyphen: `how_to_use_llvm-g++.md` vs `how-to-use-llvm-g++.md`. 
  * Spaces aren't allowed because referencing such files by name from a command-line has extra caveats, and is generally a pain if you aren't prepared.
  * If you are fond of word-wise traversal in text-editors you may prefer hyphens to underscores. Whichever is more sensible for your preferences and the content.
* Files should be named in a way that makes clear their content and role in the training scheme, but should not carry metadata. GitHub is a revision control system, so don't use filenames for versioning, ownership, or stamping date/time.
* Everything in this repository is instructional. You do not need to include qualifiers like `tutorial` or `walkthrough` in the name of the module files or directory. 

### Directory structure

Here is a brief overview of how files and directories should be organized within this repository. Explicitly listed files or directories should remain fixed in location and name, and assumed to exist as such by other files:

```bash 
HPC  # i.e. the root of this repo
├── assets
│   └── <...>  # This directory should contain all non-text files used within other markdown files.
├── template
│   └── <...>  # This is where inexperienced contributors can find a quick guide on Markdown features.
├── LICENSE.txt
├── README.md  # The homepage of the repository. Should contain a link to each module that exists below.
├── style-guidelines.md  # This is the document you're currently reading.
└── <...>      # Modules that exist or will exist
    ...
    ├─ README.md  # Any directory should contain a "README.md" to serve as the landing page.
    ...
    └ ... ─ <...>  # Sub-modules that exist or will exist
```

* Module directories should go in the root of the repository, both here and the wiki. The `README.md` is responsible for sorting the modules into a reasonably traversable hierarchy, sorted by expected user-competency and expertise with HPC principles or specific components of your module (e.g. expertise in Python).

* Per tool/software/language tutorials should have a dedicated directory and relevant source code or scripts stored within that, accompanied by a `README.md` that briefly explains how to use the content present there.

* Long-form, highly verbose documentation should be contained in an identically named directory within the [Wiki](https://github.com/NREL/HPC/wiki) of this repository. If there is content on the wiki, it should be linked there from a module that shares its name.

* **TL;DR&mdash;this repository should be predominantly composed of scripts/source code/executables or other things that _do_ something on the HPC systems, and the Wiki should predominantly be for reading about what/how/why.**

Example of a module directory:
  * In the repository itself:
    ```bash 
    plexos-quickstart
    ├── data
    │   └── RTS-GMLC-6.400.xml
    ├── env-6.4.2.sh
    ├── env-7.3.3.sh
    ├── env-7.4.2.sh
    ├── README.md  # Contains enough instruction to use these scripts. Links to the wiki for extra info.
    └── util
        └── get_week.py
    ```
  * The respective [Wiki](https://github.com/NREL/HPC/wiki) directory featuring verbose explanations:
    ```bash
    plexos-quickstart
    ├── initial-session.md
    ├── README.md  # The "home page" of the module which introduces, links to, and structures neighboring pages.
    ├── run-magma.md
    ├── run-plexos.md
    └── setup-plexos.md
    ```

### Markdown content and structure
* The intent of training or tutorial materials is to show someone who does not know how to do what you do how to do it. Err on the side of verbosity.
* Where you can assume that a user should have worked through training materials conceptually prior to the current ones, cross-reference them (mention and hyperlink). This also applies to information on the HPC website. In general, where you can refer someone to well crafted information online that's more extensive or well constructed than you have time or space to write, do so through a hyperlink.
  * **Images and other reusable non-textual files go in `assets/`** in the root of both the repository and the [Wiki](https://github.com/NREL/HPC/wiki) of this repository, so as to be easily and mutually referred to throughout various pages. GitHub doesn't support embedded videos, so if you need that please just insert it as a regular hyperlink (or hyperlinked thumbnail image of the video). If an image can replace lots of text, by all means include it. However, don't take terminal screenshots when `inline code` or a codeblock will suffice.

* As the content generation process evolves, some contributions will undoubtedly stand out as exemplary. Don't be shy about copying those in style, syntax, etc.

* Always preview the rendered output. GitHub's specific rendering has unique features and inconsistencies with other platforms.


