default:
	@cat makefile

env:
	python3 -m venv env; . env/bin/activate; pip install --upgrade pip

update: env
	. env/bin/activate; pip install -r requirements.txt

register:
	. env/bin/activate && python3 -m ipykernel install --user --name env --display-name "Python (BTC env)"

freeze:
	. env/bin/activate; pip freeze > requirements.txt

test:
	. env/bin/activate; pytest -vvx

conda_env:
	conda pytorch_env create -f environment.yml

conda_create_yaml:
	conda pytorch_env export --from-history > environment.yml

conda_update:
	conda pytorch_env update -f environment.yml --prune

conda_register:
	python3 -m ipykernel install --user --name pytorch_env --display-name "pytorch_env"

conda_makeitso: conda_env conda_register