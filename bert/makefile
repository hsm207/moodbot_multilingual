train-model:
	python finetune.py

serve-model: train-model
	uvicorn main:app --reload \
		--log-level debug \
		--port 8000 \
		--host 0.0.0.0