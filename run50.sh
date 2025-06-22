# An endless training loop, restarting the program after critical errors

if [ -d .venv ]; then
    source .venv/bin/activate
fi

while true ; do python run.py -c img-classification -e 50 -r 1; done