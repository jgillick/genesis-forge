.PHONY: all clean build deploy docs serve llms

all: build

clean:
	rm -rf dist/ build/ *.egg-info/

lint:
	uv run ruff check genesis_forge examples tests

test:
	uv run pytest -v --disable-warnings --maxfail=1

build: clean lint test
	uv build

deploy: build
	uv run twine upload dist/*

docs:
	uv pip install -r ./docs/requirements.txt
	mkdocs build
	cp dist/docs/llms.txt llms.txt
	cp dist/docs/llms-full.txt llms-full.txt

serve:
	uv pip install -r ./docs/requirements.txt
	mkdocs serve

