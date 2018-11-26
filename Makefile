.PHONY: all write latex wc hglog clean

BASENAME = thesis
CONTENT_FILES = front_matter.tex introduction.tex atomic_physics.tex numerics.tex software.tex velocimetry.tex hidden_variables.tex   

BASE_LATEX_COMMAND = lualatex -interaction=nonstopmode -halt-on-error -recorder -shell-escape

# If "pydflatex" is installed, use it to parse the logfile and print more useful output:
PYDFLATEX := $(shell command -v pydflatex 2> /dev/null)
ifdef PYDFLATEX
   PYDFLATEX_COMMAND = echo '====== BEGIN LATEX LOG ====='; pydflatex -l $(BASENAME).tex; echo '======= END LATEX LOG ======'
   LATEX_COMMAND = $(BASE_LATEX_COMMAND) $(BASENAME).tex > /dev/null; $(PYDFLATEX_COMMAND)
   LATEXMK_COMMAND = $(BASE_LATEX_COMMAND) %O %S > /dev/null; $(PYDFLATEX_COMMAND)
else
    LATEX_COMMAND = $(BASE_LATEX_COMMAND) $(BASENAME).tex
    LATEXMK_COMMAND = $(BASE_LATEX_COMMAND) %O %S
endif

# Tell TeX to just print lines as long as it wants:
export max_print_line=1048576

all: pdfannotations
	@# make with latexmk - runs latex and bibtex repeatedly as required
	@latexmk -silent -pdflatex="$(LATEXMK_COMMAND)" $(BASENAME).tex -pdf

write: pdfannotations
	@# latexmk in continuous view mode, running as necessary whenever files change
	@latexmk -silent -pvc -pdflatex="$(LATEXMK_COMMAND)" $(BASENAME).tex -pdf

latex: pdfannotations
	@# Just run latex by itself.
	@$(LATEX_COMMAND)

pdfannotations:
	@# Extract PDF annotations from the labscript paper for use with pdfpages
	@pdfannotextractor labscript_paper/labscript_paper.pdf

wc:
	@# Wordcount:
	@texcount -total $(CONTENT_FILES) | tee wc.txt

hglog:
	@# Write some info about the last commit to a file to be read into tex and printed as a footer in drafts:
	@hg log -l1 --template 'rev:     {rev} ({node|short})\nauthor:  {author}\ndate:    {date|date}\nsummary: {desc}' > hglog.txt

clean:
	@# Delete temporary files
	@rm -rf *.aux *.log *.out *.toc *.loc *.lot *.bbl *.blg *.brf *.fls *.fdb_latexmk hglog.txt _minted-$(BASENAME)/ labscript_paper/labscript_paper.pax




