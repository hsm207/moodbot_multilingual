train-model:
	python finetune.py

serve-model: train-model
	uvicorn main:app --reload --log-level debug