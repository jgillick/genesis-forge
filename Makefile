.PHONY: all clean lint test build build-forge build-runtime deploy deploy-forge deploy-runtime \
	docs serve version version-forge version-runtime check-version check-tag check-tag-forge \
	check-tag-runtime

FORGE_PYPROJECT   = packages/genesis-forge/pyproject.toml
RUNTIME_PYPROJECT = packages/genesis-forge-runtime/pyproject.toml
VERSIONS          = uv run --frozen python scripts/versions.py

all: build

clean:
	rm -rf dist/ build/ *.egg-info/

lint:
	uv run ruff check .

test:
	uv run pytest -v

# Both packages are released together by default. The -forge and -runtime
# variants of build, deploy, and version release one package on its own, for a
# patch that only touches that package.

build: clean check-version lint test
	uv build --all-packages

build-forge: clean check-version lint test
	uv build --package genesis-forge

build-runtime: clean check-version lint test
	uv build --package genesis-forge-runtime

deploy: check-tag build
	uv run twine upload dist/*

deploy-forge: check-tag-forge build-forge
	uv run twine upload dist/*

deploy-runtime: check-tag-runtime build-runtime
	uv run twine upload dist/*

docs:
	uv pip install -r ./docs/requirements.txt
	uv run mkdocs build
	cp dist/docs/llms.txt llms.txt
	cp dist/docs/llms-full.txt llms-full.txt

serve:
	uv pip install -r ./docs/requirements.txt
	uv run mkdocs serve

# Set the release version on both packages, then commit and tag it vX.Y.Z.
version:
	@$(VERSIONS) bump all "$(V)"
	uv version --frozen --package genesis-forge-runtime $(V)
	uv version --frozen --package genesis-forge $(V)
	uv add --package genesis-forge "$$($(VERSIONS) pin)"
	$(MAKE) check-version
	git add $(FORGE_PYPROJECT) $(RUNTIME_PYPROJECT) uv.lock
	git commit -m "Version $(V)"
	git tag -a "v$(V)" -m "Version $(V)"
	@echo "Tagged v$(V). Push with: git push && git push origin v$(V)"

# Patch release of genesis-forge alone, tagged genesis-forge-vX.Y.Z. Its runtime
# pin is raised to the workspace's runtime version, the one it was tested with.
version-forge:
	@$(VERSIONS) bump genesis-forge "$(V)"
	uv version --frozen --package genesis-forge $(V)
	uv add --package genesis-forge "$$($(VERSIONS) pin)"
	$(MAKE) check-version
	git add $(FORGE_PYPROJECT) uv.lock
	git commit -m "genesis-forge $(V)"
	git tag -a "genesis-forge-v$(V)" -m "genesis-forge $(V)"
	@echo "Tagged genesis-forge-v$(V). Push with: git push && git push origin genesis-forge-v$(V)"

# Patch release of genesis-forge-runtime alone, tagged genesis-forge-runtime-vX.Y.Z.
# genesis-forge's pin already accepts any patch of the same minor, so it is untouched.
version-runtime:
	@$(VERSIONS) bump genesis-forge-runtime "$(V)"
	uv version --package genesis-forge-runtime $(V)
	$(MAKE) check-version
	git add $(RUNTIME_PYPROJECT) uv.lock
	git commit -m "genesis-forge-runtime $(V)"
	git tag -a "genesis-forge-runtime-v$(V)" -m "genesis-forge-runtime $(V)"
	@echo "Tagged genesis-forge-runtime-v$(V). Push with: git push && git push origin genesis-forge-runtime-v$(V)"

# Fail unless the packages share a major.minor version, genesis-forge pins the
# runtime to it, and uv.lock is up to date. The rules are in scripts/versions.py.
check-version:
	uv sync --locked
	@$(VERSIONS) check

# Only tagged commits get published, so what is on PyPI is always something
# `git checkout <tag>` can reproduce.
check-tag:
	@$(VERSIONS) check-tag all

check-tag-forge:
	@$(VERSIONS) check-tag genesis-forge

check-tag-runtime:
	@$(VERSIONS) check-tag genesis-forge-runtime
