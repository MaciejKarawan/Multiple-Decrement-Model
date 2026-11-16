# Multiple-Decrement-Model

Install Python 3.11+ and `python -m pip install -e .`. From the repository root run `python -m scripts.run_analysis`. Commands execute experiments or launch the application; importing the model package is separate.

See [layout](docs/layout.md) and [file migration](docs/file-migration.json). Source implementations, author attribution in supplied reports and distinct historical variants are retained. New figure/table outputs are routed to results/generated. Dependency declarations describe imports, not a reconstruction of the original historical environment. No new licence is assigned.

## Current limitations

Known pre-existing blockers remain: part2 parameters are inside a disabled string block (NameError x), and mu_f uses data_m. These numerical/experiment choices were not silently changed during file organization.

## Checks

`python -m unittest discover -s tests -v` runs migration checks. Full financial experiments, agent tournaments and GUI sessions were not rerun.
