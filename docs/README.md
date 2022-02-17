
# Development

The following Docker command can be run to build a local version of the docs for development. The site will be available at localhost:8000 in your browser.

```
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs squidfunk/mkdocs-material
```
