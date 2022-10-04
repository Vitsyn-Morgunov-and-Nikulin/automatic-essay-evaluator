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


move:
	rsync -avzr ./ shamil@172.28.163.21:/home/shamil/automatic-essay-evaluator/ \
  	  --exclude '.git' --exclude '.pytest_cache' --exclude '__pycache__' --exclude '.idea' \
  	  --exclude '.env'


train:
	PYTHONPATH='.' python src/bert_featurizer_solution.py

# Calculation class metric: {'cohesion': 0.5035229486310918, 'syntax': 0.45981305537879363, 'vocabulary': 0.41329032844658387, 'phraseology': 0.48887827546824647, 'grammar': 0.5147782841966032, 'conventions': 0.4716236425673635}
  #Calculating kaggle metric: 0.47531775578144714