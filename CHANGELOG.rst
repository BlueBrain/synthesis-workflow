Changelog
=========

Version 0.0.11
--------------

- Pin versions before moving to region-grower >= 0.2
- Use importlib in setup.py instead of imp
- Import morph-validator functions, fix the tests and fix dependencies
- Create distributions for axon according to https://bbpcode.epfl.ch/code/\#/c/52107
- Make choose_morphologies export scores
- Add density map tool
- Use workflow rendering functions that were transfered to luigi-tools
- Use dir-diff-content, update to py38 and bump dependencies
- Fix compatibility with Py38
- Black the code with line length of 100
- Update tox to for py36 only for linting
- Use luigi-tools>=0.0.5 to automatically create parent directories of task targets

Version 0.0.10
--------------

Improvements
~~~~~~~~~~~~
- Add methodology in the doc
- Use luigi-tools package

Bug Fixes
~~~~~~~~~
- Fix BuildAxonMorphologies to use worker from placement_algorithm
- Fix PlotPathDistanceFits for mtypes with no fit

Version 0.0.9
-------------

New features
~~~~~~~~~~~~
- Add a task to create annotation.json file

Improvements
~~~~~~~~~~~~
- Minor doc updates

Bug Fixes
~~~~~~~~~
- Fix parallelization in vacuum synthesis
- Fix requirements

Version 0.0.8
-------------

New features
~~~~~~~~~~~~
- Add score matrix report

Improvements
~~~~~~~~~~~~
- Simplify doc and improve its generation

Version 0.0.7
-------------

Improvements
~~~~~~~~~~~~
- Add examples of configuration files into the doc

Version 0.0.6
-------------

Improvements
~~~~~~~~~~~~
- Improve doc and tests

Version 0.0.5
-------------

Improvements
~~~~~~~~~~~~
- Fix CLI to publish doc
- Update changelog for previous releases

Version 0.0.4
-------------

Improvements
~~~~~~~~~~~~
- Add doc publishing
- Add CLI for MorphVal

Bug Fixes
~~~~~~~~~~~~
- Fix CLI for synthesis-workflow

Version 0.0.3
-------------

Bug Fixes
~~~~~~~~~~~~
- Fix doc generation
- Fix requirements

Version 0.0.2
-------------

Improvements
~~~~~~~~~~~~
- Improve doc generation

Version 0.0.1
-------------
- First release
