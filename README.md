# Dev

## Adding content
The GitHub pages content is rendered from the markdown files in this repository. Each file uses standard markdown, see [GitHub Markdown Guide](https://guides.github.com/features/mastering-markdown/) for more information. 

important branches:
* auxsys: This branch is being utilized to create documentation for new systems coming online. 
* gh-pages: This branch is the production rendered site. 

The best way to contribute updates is to first [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository. Then clone your fork and add to the desired branch or create a new feature branch from the desired branch. 

### Front matter
The [front matter](https://jekyllrb.com/docs/front-matter/) is the primary method for controlling where a page is rendered. The front matter is placed at the top of each markdown file. The parent and grand_parent values let you specify hierarchy of directories and markdown. 

Example: In the below front matter `grand_parent` indicates the top level navigation on the site is Systems. Swift in the `parent` field indicates that this page is under Swift. 

```
---
layout: default
title: Running on Swift
parent: Swift
grand_parent: Systems
---
```
**The front matter must be added to each markdown file otherwise it will not be rendered.**

### Adding a new top level 
To add a new top level category you should create a new directory and within that directory a markdown file. The front matter in this markdown is similar to that for other pages. The title of this markdown will be used by other pages as the parent/grand_parent value mentioned previously. Likewise, `has_children` indicates the category will have additional pages underneath it in the navigation. 

```
---
layout: default
title: Systems
has_children: true
order: 4
---
```

### Pushing changes
Changes should be opened as Pull Requests for the branch you are adding to. 


## Running locally
The easiest way to run this GH Pages locally is using Docker. The following command will install the necessary Ruby Gems and build the site. The site will be available at `localhost:4000/HPC/`

```
docker run -it -p 4000:4000 -v "$PWD:/srv/jekyll" jekyll/builder ./build_and_serve.sh 
```