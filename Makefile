.PHONY: all clean build deploy

all: build

clean:
	rm -rf dist/ build/ *.egg-info/

build: clean
	python -m build

deploy: build
	twine upload dist/*
