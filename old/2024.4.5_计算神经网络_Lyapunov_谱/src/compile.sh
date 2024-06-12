#!/bin/bash
cd src
xelatex -output-directory=. /app/src/grad_cal.tex
bibtex /app/src/grad_cal.aux
xelatex -output-directory=. /app/src/grad_cal.tex
xelatex -output-directory=. /app/src/grad_cal.tex
