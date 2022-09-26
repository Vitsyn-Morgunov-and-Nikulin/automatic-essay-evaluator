download_data: download_core_data download_word_frequencies

download_core_data:
	mkdir -p data/raw
	cd data/raw; \
	rm *; \
	kaggle competitions download -c feedback-prize-english-language-learning; \
	unzip feedback-prize-english-language-learning.zip; \
	rm feedback-prize-english-language-learning.zip; 

download_word_frequencies:
	mkdir -p data/word_frequencies/
	cd data/word_frequencies; \
	rm *; \
	kaggle datasets download rtatman/english-word-frequency; \
	unzip english-word-frequency.zip; \
	rm english-word-frequency.zip;

test:
	flake8
	isort src/
	pytest

