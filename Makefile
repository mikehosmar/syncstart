.PHONY: man test up check dist

.PHONY: test
test:
	pytest

.PHONY: man
man:
	stpl README.rst.stpl README.rst
	pandoc README.rst -s -t man -o syncstart.1

.PHONY: check
check:
	restview --long-description --strict

.PHONY: dist
dist: man
	sudo python -m build .

.PHONY: up
up:
	twine upload dist/`ls dist -rt | tail -1` -u__token__ -p`pass show pypi.org/syncstart_api_token`

.PHONY: tag
tag: dist
	$(eval TAGMSG="v$(shell python ./syncstart.py --version | cut -d ' ' -f 2)")
	echo $(TAGMSG)
	git tag -s $(TAGMSG) -m"$(TAGMSG)"
	git verify-tag $(TAGMSG)
	git push origin $(TAGMSG) --follow-tags


