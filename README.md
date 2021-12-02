# About
This branch includes the content rendered on the  nrel.github.io/hpc site. The focus of this site is to provide documentation and blog posts to enable users to utilize NREL's computational resources. 

For code tutorials and workshops see the master branch of this repository. 

## Development

The following Docker command can be run to build a local version of the docs for development. The site will be available at `localhost:8000` in your browser. 
```
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs squidfunk/mkdocs-material
```