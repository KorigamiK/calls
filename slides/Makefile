.PHONY: presentation-pdf presentation-pptx presentation-html

presentation-html:
	pandoc --defaults=./defaults.yaml --defaults=html.yaml --katex=./katex/ --reference-location=section
	xdg-open file://$(shell pwd)/presentation.html?print-pdf

presentation-pdf:
	pandoc --defaults=./defaults.yaml --defaults=pdf.yaml

presentation-pptx:
	pandoc --defaults=./defaults.yaml --defaults=pptx.yaml
