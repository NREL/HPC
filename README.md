# Dev

## Running locally
The easiest way to run this GH Pages locally is using Docker. The following command will install the necessary Ruby Gems and build the site. The site will be available at `localhost:4000/HPC/`

```
docker run -it -p 4000:4000 -v "$PWD:/srv/jekyll" jekyll/builder ./build_and_serve.sh 
```