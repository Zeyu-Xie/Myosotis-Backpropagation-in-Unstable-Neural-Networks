#!/bin/bash
xelatex -output-directory=. /app/thuthesis-example.tex
openout_any=a bibtex /app/thuthesis-example.aux
xelatex -output-directory=. /app/thuthesis-example.tex
xelatex -output-directory=. /app/thuthesis-example.tex