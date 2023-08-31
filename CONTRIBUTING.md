# Contributing

Thank you for taking an interest in contributing to our repository! Please consult the information below (we tried to keep it brief and reasonable) prior to making a pull request. These guidelines are designed to help avoid disputes between the repository maintainers and contributors by making the standards as obvious and transparent as possible.

If you desire, you may credit yourself as author within any content you contribute.

### Who can contribute?
Can you read this? Then you! Simply consult the short [style guidelines](#style-guidelines) below to get familiar with our content standards.

### What should be contributed?
* Tutorials on your software-stack of choice (something you'd give a new member of your team)
* Scripts/notebooks/cheatsheets that automate a convoluted task or explain something succinctly
* Explanations of system aspects of Linux and specific NREL HPC systems (e.g. "What is the `/var` directory for?")
* Anything you think another HPC user might benefit from!

### How do I contribute?

If the below instructions are foreign to you, consider reviewing the [`git` module](/general/beginner/git/) to "*git*" familiar with git! The git workflow has some colorful jargon but none of the concepts should be new to you. For the absolute beginner (unfamiliar with Linux or the command line interface), then it might be best to [start somewhere more basic](/general/beginner/intro-to-linux/).

#### Where do I put my content? 
There are two primary locations for contributions: etiher this base repository, or the gh-pages branch, which renders files on our [Github Pages website](https://nrel.github.io/HPC/). The gh-pages branch is for contributions that require more explanations and/or are highly verbose, whereas this repo should be predominantly composed of scripts/source code/executables that do something on the HPC systems.

1. <a href="https://github.com/NREL/HPC/fork">Fork this repo <img src="https://img.shields.io/github/forks/NREL/HPC.svg?style=social"></a> 
1. _(optional)_ To edit locally on your machine, do either of:
   * `git clone https://github.com/`\<your GitHub username\>`/HPC` for only the base repo.
1. If contributing to gh-pages:
   * `git checkout gh-pages`
1. Change something (_after_ consulting the [style guidelines](#style-guidelines))
1. `git add` the change(s)
1. `git commit` with a useful commit message!
1. `git push`
1. Make a pull-request (shiny green button on your fork of the repo's GitHub webpage.)

Alternatively, you may send your contributions via e-mail attachment to HPC-Help@nrel.gov with subject of "NREL HPC GitHub Contribution" and the body detailing what changes you made (the more specific the better).

### Why should I contribute?
We love collaboration, and your contributions add value to an open, searchable HPC knowledge-base that is usable not only by the NREL HPC community, but also by the HPC community at large. Thank you for your efforts in sharing knowledge with the world!

---

## Style Guidelines

### ***New to Markdown?***
Appropriately enough, we have documentation for that! Simply start with the [README in the Markdown module](/general/beginner/markdown/README.md). Not to mention, the raw-contents of any `.md` file can be used as a reference for how to style content a certain way.

### Markdown content and structure
* The intent of training or tutorial materials is to show someone who does not know how to do what you do how to do it. Err on the side of verbosity.

* Where you can assume that a user should have worked through other training materials or concepts prior to your contribution, please cross-reference the other material (with a mention and a hyperlink). This also applies to information on our HPC website. In general, where you can refer someone to well-crafted information online that's more extensive or well-constructed than you have time or space to write, do so through a hyperlink.

### General Advice

* As the content generation process evolves, some contributions will undoubtedly stand out as exemplary. Don't be shy about copying those in style, syntax, etc.

* Always preview the rendered output. GitHub's specific rendering has unique features and inconsistencies with other platforms.

## Files and directories

### Naming files
* Files need an appropriate extension so it is obvious what purpose they serve. Compiled binaries shouldn't be pushed here (the most common type of extension-lacking files) so there's no reason a file should be without an extension.
  * This includes Markdown files, which should carry a ".md" extension (not .txt or lackthereof).
* No capitals or spaces (with the exception of any `README.md` files serving as landing pages within directories per GitHub markdown rendering). Separate words in the filename with underscores or hyphens&mdash;whichever reduces ambiguity the most and is most pleasant for you to type.
  * For example, using underscores in a potential filename containing a hyphenated command like `llvm-g++` would make it clear that the command has a hyphen: `how_to_use_llvm-g++.md` vs `how-to-use-llvm-g++.md`. 
  * Spaces aren't allowed because referencing such files by name from a command-line has extra caveats, and is generally a pain if you aren't prepared.
* Files should be named in a way that makes clear their content and role in the training scheme, but should not carry metadata. GitHub is a revision control system, so don't use filenames for versioning, ownership, or stamping date/time.
* Everything in this repository is instructional. You do not need to include qualifiers like `tutorial` or `walkthrough` in the name of the module files or directory. 

The following content is specific to the base repository, please see the [Github Pages section](#github-pages) for contribution information unique to that branch.

### Directory structure 

Here is a brief overview of how files and directories should be organized within this repository. Explicitly named files or directories should remain fixed in location and name, and assumed to exist as such by other files:

```bash 
HPC  # i.e. the root of this repo
├── assets
│   ...
│   └── <...>  # This directory should contain all non-text files used within other markdown files.
├── CONTRIBUTING.md  # This is the document you're currently reading.
├── LICENSE.txt
├── README.md  # The homepage of the repository. Should contain a link to each module's README.
...
└── <...>      # Modules that exist or will exist
    ...
    ├─ README.md  # Any directory should contain a "README.md" to serve as the landing page.
    ...
    └ ... ─ <...>  # Sub-modules that exist or will exist
```

* Module directories should go in the root of the repository, both here and the wiki. The `README.md` is responsible for sorting the modules into a reasonably traversable hierarchy, sorted by expected user-competency and expertise with HPC principles or specific components of your module (e.g. expertise in Python).

* Per tool/software/language tutorials should have a dedicated directory and relevant source code or scripts stored within that, accompanied by a `README.md` that briefly explains how to use the content present there.

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
* **Images and other reusable non-textual files go in `assets/`** in the root of the repository to be easily and mutually referred to throughout various pages. The image can then be linked to anywhere with markdown as `![alt text](/assets/name_of_image.png)`. If an image can replace lots of text, by all means, please include it. However, don't take terminal screenshots when `inline code` or a codeblock will suffice.
  
*  GitHub doesn't support embedded videos, so if you need that please just insert it as a regular hyperlink (or hyperlinked thumbnail image of the video). 

### Github Pages

Here is a brief overview of how files and directories should be organized. Explicitly named files or directories should remain fixed in location and name, and assumed to exist as such by other files:

```bash 
HPC  # i.e. the root of this repo
├── docs
    ...
    └── Documentation      # This directory is where documentation contributions should go. 
        ...
        └── <...>      # Modules that exist or will exist
            ...
            ├─ index.md  # Optional overview page for the content section.
            ...
            └ ... ─ <...>  # Sub-modules that exist or will exist.
├── overrides  # This directory should contain images.
├── mkdocs.yml # This file contains configuration settings for the site.
├── README.md  # The homepage of the repository.
```
* If your documentation fits into one of the categories currently in the pages, you may place it in the coresponding directory. Otherwise, make a new directory in Documentation. 
* **Files must be listed in the mkdocs.yml nav section in order to be rendered.** The nav section dictates the navigation bar structure on the site. Section headers must be included followed by an indented list of the paths to the files the section should contain. Please see example below. 
```bash 
 - Documentation:
      - Data and File Systems:
        - File Systems:
          - Documentation/Data-and-File-Systems/File-Systems/index.md
          - Lustre:
            - Documentation/Data-and-File-Systems/File-Systems/Lustre/lustrebestpractices.md
```          
* If you would like the section header in the navigation bar to be directly linked to documentation (e.g. an overview page), place this content in an index.md file in the respective folder, and add it to the beginning of its nav section in the mkdocs.yml file. 
* Images should be placed in the overrides directory and linked to with the relative path without the overrides directory included like ```../../../assets/images/conda_logo.png``` **not** ```../../../overrides/assets/images/conda_logo.png```.

*  Any links to internal files must be the relative path to it from the referencing file, not the absolute path. 
*  Code blocks will be automatically highlighted when rendered on the website, so please either use a code block or `inline` code whenever practical instead of a screenshot of a terminal.

## Blog posts

### Creating a blog post
Here is a brief overview on how to create and style a new blog post:

First, we need to create the Markdown file for the blog. The files name must follow the following styling. The files name must start with the date in this format XXXX-XX-XX- where each of the X sections correspond to this format YEAR-MONTH-DAY- and then is followed by a simple name that has something to do with the blog. Example Markdown file name: 2023-12-02-python_tutorial.md. The file must be placed within the posts directory which can be found as follows: 

```bash 
 - HPC
  - docs
    - blog
      - posts
        - XXXX-XX-XX-new_file_goes_in_here.md
```  

Second, we need to add the author to the authors.yml file within the blog directory. Each author has four required elements. Their's the authors_name, name, description, and avatar. The authors_name is going to be the author's fullname in all lowercase with no space (example: johndoe) and this will be referenced within the blogs Markdown file later. The name is the authors full name. The description must be set to "Author" and the avatar must be set to http://nrel.github.io/HPC/blog/assets/avatar/default.png as follows:

```bash
  johndoe: # This is the authors_name 
    name: John Doe
    description: Author  
    avatar: http://nrel.github.io/HPC/blog/assets/avatar/default.png
```

Lastly, we have to format the blog post as follows:

```bash
  ---
  date: XXXX-XX-XX
  authors:
      - authors_name # This is the authors_name we created earlier within the authors.yml file
  ---

  # Title of the blog goes here with the "#" character included

  This will be the first paragraph of the blog post. This first paragraph will be previewed on the blog landing page. Try to make this first paragraph within three sentences or so to not clutter the blog landing page. Do not forget to add the "<!-- more -->" after this first paragraph as this is what will seperate the first paragraph in the blog preview from the rest of the blog. 

  <!-- more --> # This seperates the blog preview from the rest of the blog

  This is where the rest of the blog content will go.

```