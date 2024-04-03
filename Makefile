.PHONY: man test up check dist

test:
	pytest

man:
	pip install --user -e . &>/dev/null || true
	stpl README.rst.stpl README.rst
	pandoc README.rst -s -t man -o syncstart.1

check:
	restview --long-description --strict

dist: man
	sudo python -m build .

up:
	twine upload dist/`ls dist -rt | tail -1` -u__token__ -p`pass show pypi.org/syncstart_api_token`


