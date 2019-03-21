# Style Guidelines

***New to markdown?*** Just take a look at the raw-text of [`template/README.md`](template/README.md) as a starting point for what syntaxes get rendered in what ways. Any `.md` file here can be used as a reference for how to style content.

#### Naming files
* No capitals or spaces (with the exception of any `README.md` files serving as landing pages within directories per GitHub markdown rendering). Separate words in the filename with underscores or hyphens&mdash;whichever reduces ambiguity the most and is most pleasant for you to type. For example, a potential filename containing a hyphenated command `llvm-g++`. Underscores make it clear that the command has a hyphen: `how_to_use_llvm-g++.md` vs `how-to-use-llvm-g++.md`. 
  * Spaces aren't allowed because referencing such files by name from a command-line has extra caveats, and is generally a pain if you aren't prepared.
  * If you are fond of word-wise traversal in text-editors you may prefer hyphens more often. Whichever is more sensible for your tastes and the content.
* Markdown files carry a ".md" extension.
* Files should be named in a way that makes clear their content and role in the training scheme, but should not carry metadata. GitHub is a revision control system, so don't use filenames for versioning, ownership, or stamping date/time.

#### Content organization and flow
* The intent of training or tutorial materials is to show someone who does not know how to do what you do how to do it. Err on the side of verbosity.
* Where you can assume that a user should have worked through training materials conceptually prior to the current ones, cross-reference them (mention and hyperlink). This also applies to information on the HPC website. In general, where you can refer someone to well crafted information online that's more extensive or well constructed than you have time or space to write, do so through a hyperlink.
  * **Images and other reusable non-textual files go in `assets/`** in the root of both the repository and the [Wiki](https://github.com/NREL/HPC/wiki) of this repository, so as to be easily and mutually referred to throughout various pages. GitHub doesn't support embedded videos, so if you need that please just insert it as a regular hyperlink (or hyperlinked thumbnail image of the video). If an image can replace lots of text, by all means include it. However, don't take terminal screenshots when `inline code` or a codeblock will suffice.

* As the content generation process evolves, some contributions will undoubtedly stand out as exemplary. Don't be shy about copying those in style, syntax, etc.

* Double-check the rendered output. I have sometimes found that claims of what Markdown does are often wrong, out-of-date, or don't apply specifically to GitHub.

### Directory structure

* Modules directories should go in the root of the repository, both here and the wiki. The `README.md` is responsible for sorting the modules into a reasonably traversable hierarchy, sorted by expected user-competency and expertise with HPC principles or specific components of your module (e.g. expertise in python).

* Per application/language/objective tutorials should have a dedicated directory and relevant source code or scripts stored within that, accompanied by a `README.md` briefly walking through how to interact with the content present there.
  * Long-form, highly verbose documentation should be contained in the relevant section of the [Wiki](https://github.com/NREL/HPC/wiki) of this repository, and referred to within the README of the respective content directory.

    Example:
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
    and the corresponding [Wiki](https://github.com/NREL/HPC/wiki) directory featuring thorough, verbose walkthroughs and explanations:
    ```bash
    plexos-quickstart
    ├── initial-session.md
    ├── README.md  # The "home page" of the module which introduces, links to, and structures neighboring pages.
    ├── run-magma.md
    ├── run-plexos.md
    └── setup-plexos.md
    ```
