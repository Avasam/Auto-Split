echo "`nRunning Pylint..."
pylint --score=n --output-format=text,colorized $(git ls-files '**/*.py*')
# pylint --reports=y --output-format=text,colorized $(git ls-files '**/*.py*')

echo "`nRunning Pyright..."
pyright

echo "`nRunning Bandit..."
bandit -f custom --silent --severity-level medium -r .
# bandit -n 1 --severity-level medium -r src

echo "`nRunning Flake8..."
flake8
