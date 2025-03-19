default:
	@cat makefile

env:
	python3 -m venv env; . env/bin/activate; pip install --upgrade pip

update: env
	. env/bin/activate; pip install -r requirements.txt

register:
	python3 -m ipykernel install --user --name env --display-name "Python (CCO env)"

freeze:
	. env/bin/activate; pip freeze > requirements.txt

test:
	. env/bin/activate; pytest -vvx