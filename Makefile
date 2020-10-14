bump-major: PART = major
bump-minor: PART = minor
bump-patch: PART = patch

bump-major bump-minor bump-patch:
	@./bump_version.sh $(PART)

publish:
	@./publish.sh
