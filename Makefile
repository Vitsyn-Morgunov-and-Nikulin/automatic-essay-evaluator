build: download_data download_weights run

download_data:
	mkdir -p data/raw
	cd data/raw; \
	rm *; \
	kaggle competitions download -c feedback-prize-english-language-learning; \
	unzip feedback-prize-english-language-learning.zip; \
	rm feedback-prize-english-language-learning.zip

test:
	flake8
	isort .
	pytest  -p no:cacheprovider

download_weights:
	cd ./demo; \
	kaggle datasets download -d alukaevdanis/feedback-prize-weights; \
	unzip feedback-prize-weights.zip; \
	rm feedback-prize-weights.zip

run:
	PYTHONPATH=. streamlit run demo/app.py
