## Build Documentation:



#### Install Requirements

```python
pip install -r requirements.txt
```



#### Build Documentation

```bash
# Enter docs folder.
cd docs
# Use sphinx autodoc to generate rst.
sphinx-apidoc -o source/ ../matchzoo/
# Generate html from rst
make clean
make html
```
This will install all the packages need in the code. This can cause some error [issue](https://github.com/readthedocs/readthedocs.org/issues/5882)
That is not necessary.

So , we have a new way to generate documents  
Follow this [link](https://sphinx-autoapi.readthedocs.io/en/latest/tutorials.html)
```bash
pip install sphinx-autoapi
```
then modify the conf.py
```bash
extensions = ['autoapi.extension']
autoapi_dirs = ['../mypackage']
```
then
```bash
make html
```
