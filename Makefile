.PHONY: build up down notebook audit train

# Build the docker image
build:
	docker compose build

# Start the container with Jupyter running (visit localhost:8888)
up:
	docker compose up

# Stop the container
down:
	docker compose down

# Run the data audit on a session directory from the host, no Jupyter needed
# Usage: make audit SESSION=data/raw/<rat>/<session>
audit:
	docker compose run --rm seqmem python -m src.data_loading $(SESSION)

# Run training
train:
	docker compose run --rm seqmem python -m src.train --config configs/baseline.yaml
