PAPER=mikenet

preview:	
	make ${PAPER}.dvi
	xdvi ${PAPER}

dvi:	
	make ${PAPER}.dvi

quick:
	latex ${PAPER}
	xdvi ${PAPER}

${PAPER}.dvi:	${PAPER}.tex
	latex ${PAPER}
	bibtex ${PAPER}
	latex ${PAPER}
	latex ${PAPER}

print:	${PAPER}.ps 
	lpr ${PAPER}.ps

ps:	${PAPER}.ps 


${PAPER}.ps:	${PAPER}.dvi 
	dvips ${PAPER}

clean:
	rm -f *.dvi ${PAPER}.ps *~ *.log *.aux *.bbl *.blg
