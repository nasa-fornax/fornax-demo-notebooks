# The Fornax Initiative

NASA Astrophysics is developing the Fornax Initiative, a cloud-based system that
brings together data, software, and computing so that researchers can focus on science.


## User Documentation

```{toctree}
---
maxdepth: 2
caption: User Documentation
---

documentation/README

```

## Tutorial Notebooks

```{toctree}
---
maxdepth: 2
caption: Tutorial Notebooks
---

forced_photometry/README
light_curves/README
```

Note that we store these notebooks as markdown files.  In the Fornax Science Console, these should open automatically in the runnable jupyter notebook format.  If they do not, or you run them elsewhere, you may need to convert them explicitly with 

`jupytext --to notebook yourfile.md`
