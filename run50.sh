# An endless training loop, restarting the program after critical errors

source .venv/bin/activate
while true ; do python run.py -c img-classification -e 50 -r 1; done