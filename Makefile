.PHONY: build test clean

build:
	go build -o bin/repo-probe ./...

test:
	go test ./...

clean:
	rm -rf bin/
