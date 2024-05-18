#!/bin/bash
cd src
xelatex -output-directory=. /app/src/main.tex
bibtex /app/src/main.aux
xelatex -output-directory=. /app/src/main.tex
xelatex -output-directory=. /app/src/main.tex
