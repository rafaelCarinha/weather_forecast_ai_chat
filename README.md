To manually create a virtualenv on MacOS and Linux:

```
$ python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

pip install llama-cpp-python
### Run the APP 
```
chainlit run app.py -w --port 8080 # -w flag is for restarting app after each change!
```