# IgCraft: A generative model for paired antibody sequences

To setup the virtual environment use the `environment.yml` file. We recommend using `mamba`. If you don't have
`mamba` installed, install it in your **base environment** using the following command:

```bash
conda install conda-forge::mamba
```

Then create the virtual environment using the following command:

```bash
mamba env create -f environment.yml
```

Finally you'll need to `pip install` the package itself after activating the environment:

```bash
conda activate igcraft
pip install -e .
```

