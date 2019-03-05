# NREL HPC Community Repository

## Who can contribute?
Can you read this? Then you! Or anyone who's itching. Simply consult the short [style guidelines](#style-guidelines) below to get familiar with our content standards (an old college try is probably sufficient).

## What needs contributed?
First off, go ahead and fork this guy for any addition or correction you're keen on making. If you see something that's inaccurate or ambiguous, go ahead and correct. Is there something you often find yourself repeating in e-mails or meetings you'd like to finally commit to a good write-up and just share the link? Draft it up! It' not just Software Developers who need to keep DRY (**D**on't **R**epeat **Y**ourself).

## How do I contribute?
1. Fork
2. Change something (_after_ consulting the [style guidelines](#style-guidelines)&mdash;they're reasonable!)
3. `git add` the change(s)
4. `git commit` with a useful commit message!
5. `git push`
6. Make a pull-request (shiny green button on the repo's GitHub webpage.)
7. (_optional_) Delete the fork once your changes are integrated if you like to keep a minimal repository ownership.

## Why should I contribute?
Something something good Samaritan, benefit the community, searchable knowledge-base, etc. 

---

## Style Guidelines

### Under development! Just don't put PDFs in here please. Unless you really really need to.
### Training module structure
* Per application/language/objective tutorials should have a dedicated directory and relevant source code or scripts stored within that, accompanied by a `README.md` briefly walking through how to interact with the content present there.
  * Long-form, highly verbose documentation should be contained in the relevant section of the Wiki of this repository, and referred to within the README of the respective content directory.

  * Example:
    ```bash 
    plexos-hpc-walkthrough
    ├── data
    │   └── RTS-GMLC-6.400.xml
    ├── README.md  # Contains enough instruction to use these scripts. Links to Wiki for extra info.
    ├── env-6.4.2.sh
    ├── env-7.3.3.sh
    ├── env-7.4.2.sh
    └── util
        └── get_week.py
    ```
    and the corresponding Wiki pages featuring verbose explanations and linking to each other:
    ```bash
    plexos-hpc-walkthrough
    ├── INSTALL-GIT-BASH.md
    ├── Initial-Session.md
    ├── Login-HPC.md
    ├── Obtain-Account.md
    ├── Run-Magma.md
    ├── Run-PLEXOS.md
    └── Setup-PLEXOS.md
    ```
* Images and similar modular content go in `assets/` in the root of both the repository and the Wiki of this repository, so as to be easily and mutually referred to throughout various pages. GitHub doesn't support embedded videos, so if you need that please just insert it as a regular hyperlink.

* Modules directories should go in the root of the repository, both here and the wiki. The `README.md` is responsible for sorting the modules into a reasonably traversable hierarchy, sorted by expected user-competency and expertise with HPC principles or specific components of your module (e.g. expertise in python).
