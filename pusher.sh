#!/bin/bash
black "." && isort .
# create scripts if there are any notebook files
for i in $(fdfind --glob "*.ipynb" src/); do jupytext --to py $i; done
if [ ! -z $1 ]; then
        git add . && git commit -m $1 && git push
fi
