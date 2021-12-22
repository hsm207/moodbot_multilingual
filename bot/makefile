validate:
	rasa data validate

train: validate
	rasa train -vv

run-rasa-server: train
	rasa run \
		--enable-api \
		-vv \
		--cors "*"

run-action-server:
	python -m debugpy \
		--listen 0.0.0.0:5678 \
		-m rasa_sdk \
		--actions actions \
		--auto-reload \
		--debug