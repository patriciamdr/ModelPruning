INTRODUCTION
------------

Repository for the project "Model Pruning" of the practical lab "Learning for Self-Driving Cars and Intelligent Systems".

 * Weekly progress reports can be found in folder reports/ (i.e. week1, week2, ...)


REQUIREMENTS
------------

I provided a requirements.txt with all necessary packages.

The virtualenv package is required to create virtual environments. You can install it with pip:
```
pip install virtualenv
```

INSTALLATION
------------
Installation via virtualenv:


 * Create virtualenv with name venv:
    ```
    virtualenv venv
    ```

* Activate virtualenv:
    ```
    source venv/bin/activate
    ```

 * Install requirements:
    ```
    pip install -r requirements.txt
    ```

 * If you want to install other packages simply run:
    ```
    pip install packagename
    ```
 * To decativate the virtual environment and use your original Python environment, simply type ‘deactivate’.
    ```
    deactivate
    ```

FETCH DATA:
------------

I used git lfs to track large files like checkpoints and pngs. In order to execute scripts, which rely on saved models, you need to fetch all the data beforehand using:
```
git lfs fetch && git lsf checkout
```
