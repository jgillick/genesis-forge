.PHONY: all clean build deploy docs serve llms

all: build

clean:
	rm -rf dist/ build/ *.egg-info/

lint:
	uv run ruff check genesis_forge examples tests deploy

test:
	uv run pytest -v

build: clean lint test
	uv build --all-packages

deploy: build
	uv run twine upload dist/*

docs:
	uv pip install -r ./docs/requirements.txt
	uv run mkdocs build
	cp dist/docs/llms.txt llms.txt
	cp dist/docs/llms-full.txt llms-full.txt

serve:
	uv pip install -r ./docs/requirements.txt
	uv run mkdocs serve

