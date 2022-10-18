download_data:
	mkdir -p data/raw
	cd data/raw; \
	rm *; \
	kaggle competitions download -c feedback-prize-english-language-learning; \
	unzip feedback-prize-english-language-learning.zip; \
	rm feedback-prize-english-language-learning.zip


test:
	flake8
	isort src/
	pytest
