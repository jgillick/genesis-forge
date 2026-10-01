.PHONY: all clean lint test build deploy docs serve version check-version check-tag

FORGE_PYPROJECT   = packages/genesis-forge/pyproject.toml
RUNTIME_PYPROJECT = packages/genesis-forge-runtime/pyproject.toml

all: build

clean:
	rm -rf dist/ build/ *.egg-info/

lint:
	uv run ruff check .

test:
	uv run pytest -v

build: clean check-version lint test
	uv build --all-packages

deploy: check-tag build
	uv run twine upload dist/*

docs:
	uv pip install -r ./docs/requirements.txt
	uv run mkdocs build
	cp dist/docs/llms.txt llms.txt
	cp dist/docs/llms-full.txt llms-full.txt

serve:
	uv pip install -r ./docs/requirements.txt
	uv run mkdocs serve

# Set the release version on both packages, then commit and tag it.
version:
	@test -n "$(V)" || { echo "usage: make version V=x.y.z"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "Working tree is not clean; commit or stash first."; exit 1; }
	@! git rev-parse -q --verify "refs/tags/v$(V)" >/dev/null || { echo "Tag v$(V) already exists."; exit 1; }
	uv version --frozen --package genesis-forge-runtime $(V)
	uv version --frozen --package genesis-forge $(V)
	uv add --package genesis-forge "genesis-forge-runtime==$(V)"
	$(MAKE) check-version
	git add $(FORGE_PYPROJECT) $(RUNTIME_PYPROJECT) uv.lock
	git commit -m "Version $(V)"
	git tag -a "v$(V)" -m "Version $(V)"
	@echo "Tagged v$(V). Push with: git push && git push origin v$(V)"

# Fail unless both packages are the same version and genesis-forge-runtime
# is pinned to the exact version in genesis-forge's lock file.
check-version:
	uv sync --locked
	@forge=$$(uv version --package genesis-forge --short); \
	runtime=$$(uv version --package genesis-forge-runtime --short); \
	pin=$$(uv pip tree --package genesis-forge --show-version-specifiers --depth 1 \
		| grep 'genesis-forge-runtime ' \
		| grep -o '==[^]]*' \
		| tr -d '='); \
	if [ "$$forge" != "$$runtime" ]; then \
		echo "Version mismatch: genesis-forge=$$forge genesis-forge-runtime=$$runtime"; \
		echo "Run 'make version V=x.y.z' to set them all."; \
		exit 1; \
	fi; \
	if [ "$$forge" != "$$pin" ]; then \
		echo "genesis-forge pins genesis-forge-runtime==$${pin:-<none>}, expected ==$$forge, in $(FORGE_PYPROJECT)"; \
		echo "Run 'make version V=x.y.z' to set them all."; \
		exit 1; \
	fi; \
	echo "Version $$forge"

# Only tagged commits get published, so what is on PyPI is always something
# `git checkout vX.Y.Z` can reproduce.
check-tag:
	@forge=$$(uv version --package genesis-forge --short); \
	tag=$$(git describe --tags --exact-match 2>/dev/null); \
	if [ "$$tag" != "v$$forge" ]; then \
		echo "HEAD is not tagged v$$forge (found '$${tag:-no tag}'). Run 'make version V=x.y.z' first."; \
		exit 1; \
	fi
