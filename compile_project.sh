#!/bin/bash
# create scripts if there are any notebook files
for i in $(fdfind --glob "*.ipynb" src/); do jupytext --to py $i; done
