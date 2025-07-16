DOC = main

.PHONY: all tex clean

all: tex

tex:
	@latexmk -pdf $(DOC).tex

clean:
	@latexmk -c
	@rm -f *.aux
	@rm -f *.bbl
