.PHONY: thesis clean

latex_command = lualatex -file-line-error -interaction=nonstopmode -halt-on-error -shell-escape

# Tell TeX to just print lines as long as it wants:
export max_print_line=1048576

thesis:
	latexmk -pdflatex="$(latex_command)" -pdf thesis.tex

write:
	# Like "make thesis" except continually updates when files change. Good for when writing.
	latexmk -pvc -pdflatex="$(latex_command)" -pdf thesis.tex

latex:
	$(latex_command) thesis.tex

clean:
	rm -rf *.aux *.log *.out *.toc *.bbl *.blg *.brf *.fls *.fdb_latexmk _minted-thesis/




