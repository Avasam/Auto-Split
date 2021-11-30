echo "`nRunning Pyright..."
pyright

echo "`nRunning Pylint..."
pylint --score=n --output-format=colorized $(git ls-files '**/*.py*')

echo "`nRunning Bandit..."
bandit -f custom --silent --severity-level medium -r .

echo "`nRunning Flake8..."
flake8
