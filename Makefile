.PHONY: clean
latex_command = lualatex -file-line-error -interaction=nonstopmode -halt-on-error -shell-escape thesis.tex
bibtex_command = bibtex thesis.aux

clean:
	rm -rf *.aux *.log *.out *.toc *.bbl *.blg *.brf _minted-thesis/

latex:
	$(latex_command)

bibtex:
	$(bibtex_command)

full: clean
	$(latex_command)
	$(bibtex_command)
	$(latex_command)
	$(latex_command)

