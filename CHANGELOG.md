# Changelog

## [synthesis-workflow-v1.0.0](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.1.3...synthesis-workflow-v1.0.0)

> 15 May 2023

### New Features

- Add region support (Alexis Arnaudon - [506be99](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/506be99164d017cd9a1060dcb443e2d6c910ae6f))

## [synthesis-workflow-v0.1.3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.1.2...synthesis-workflow-v0.1.3)

> 15 May 2023

### Build

- Set max version for region-grower (Adrien Berchet - [b11e682](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/b11e682b2e672d9a2463899aa9baaa57bf987f71))

### Fixes

- Axon fixing (Alexis Arnaudon - [fd5b184](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/fd5b18438de931dffa8cc98cb6c98bd4afda7891))

### General Changes

- Fix circuit place seed (Alexis Arnaudon - [be61217](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/be61217cb90b533c4103ac197afa941607a2412d))

## [synthesis-workflow-v0.1.2](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.1.1...synthesis-workflow-v0.1.2)

> 14 March 2023

### New Features

- Use 3d_angles and simpler diametrizer by default (Alexis Arnaudon - [14d6a2d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/14d6a2d1b5c9e870d98abf52c68bda7ff30575ed))

### Chores And Housekeeping

- Fix for numpy&gt;=1.24 and use more optional parameters (Adrien Berchet - [3b53020](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/3b53020463bb63feb5a47a5e1ed1cdcdd79743b6))
- Update from copier template (Adrien Berchet - [50873ab](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/50873abcf36803458dbca0401d561c0c868e7479))
- Bump brainbuilder (Adrien Berchet - [9c3849f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/9c3849fd4db83d8df89a3d4a363dadab87b47c72))
- Fix lint (Adrien Berchet - [f19a3a3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f19a3a3d1de47432605837358df4a07523663b25))
- Apply Copier template (Adrien Berchet - [d6dcc09](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d6dcc09226a581c7b5835a812d7c08ef19a2c59f))
- Add JSON schemas to ListParameter objects (Adrien Berchet - [03a48db](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/03a48db30e02f95b9e0eb50b48146a2773767456))
- Restrict brainbuilder to !=0.18.1 because of an issue with 56f304fe46a4d1c3ea14460eab735a20fc3ae056 (Adrien Berchet - [1433620](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/143362021ae07240d19ebabcfd72dc48183c9a07))

### Documentation Changes

- Fix indent in lists (Adrien Berchet - [9815ee8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/9815ee8068bae75afc6af22bfe1b1af92894500a))

### Refactoring and Updates

- Apply Copier template (Adrien Berchet - [61ee764](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/61ee7645f924218eb5a5ffb7592f951cacece8b5))
- Update from template (Adrien Berchet - [1cc8e4d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1cc8e4d64c275873206818a70d6d9e0b6852b14a))

### CI Improvements

- Bump pre-commit hooks and fix isort installation failure (Adrien Berchet - [c9996c2](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c9996c2902211963cddb02998bfff691dcf70f36))
- Fix coverage for pytest-cov&gt;=4 (Adrien Berchet - [cf305d3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/cf305d3124da0d756a44503d038eb4a5885e5147))
- Fix tests for new versions of Pandas and Sphinx (Adrien Berchet - [ecd8dba](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/ecd8dbad84484832cf325d5960bfa9f7463804cf))

### General Changes

- Use external file for custom parameters (Alexis Arnaudon - [4bcebcc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/4bcebcc63cf90f952b26894bb1a3425035a80130))
- Use NeuroCollage (Alexis Arnaudon - [5799ac7](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/5799ac7a8453dcff836cf53baf2064063b7f8e1b))
- Use latest diameter-synthesis (Alexis Arnaudon - [d0eaafd](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d0eaafda25b4a2fe7be8b7fbe3b1ff0e1a23daff))
- Export circuitconfig (Alexis Arnaudon - [44db085](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/44db085867fc6b46b7694f9e30b52bcca10bfcb9))
- Fix substitution rule (Alexis Arnaudon - [5488996](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/548899663b26c989c53ebad5e18ab2662933a763))

## [synthesis-workflow-v0.1.1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.1.0...synthesis-workflow-v0.1.1)

> 5 July 2022

### General Changes

- Any region handling (v2) (Alexis Arnaudon - [3fd475a](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/3fd475a6f7711b21d04e989959cf6bd05528b4fa))
- Trunk validation (Alexis Arnaudon - [b0b407e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/b0b407e2ead89803053ccdf264d9993c145f8125))
- Bump luigi-tools (Adrien Berchet - [9a19543](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/9a1954344f41ff5bc31d444b16d0fd4fecc62746))
- Add commitlint on MR titles (Adrien Berchet - [0f1e9d4](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/0f1e9d448002c5348ceb00c7736994f332862e98))
- Fix doc generation (Adrien Berchet - [daf066f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/daf066f48ae4b7d4988bcf5652dbf981ca4afbfb))
- Add multi-project trigger to ensure SynthDB is compatible with each new tag (Adrien Berchet - [15ec479](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/15ec4798e9f575b5946bdde6b86f78a41cd1fc9f))
- Fix the multi-project trigger (Adrien Berchet - [95b95a3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/95b95a35ee5c2d66029d44e7575310d6f45f894b))

<!-- auto-changelog-above -->

## [synthesis-workflow-v0.1.0](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.11...synthesis-workflow-v0.1.0)

> 1 April 2022

- Insitu validation (collage + extents plots) (arnaudon - [d3e611d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d3e611d533ab4e00e00fddf73d40dc0af4aff037))
- Use new versions of region-grower, TNS and diameter-synthesis (Adrien Berchet - [06257d7](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/06257d71e7084f71809b5e8b916a859457882aef))
- Migrate from Gerrit to GitLab (Adrien Berchet - [e14ecc1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e14ecc113405f88cbf9bf31244b13f34c56afa88))
- Remove global-exclude from MANIFEST (Adrien Berchet - [17bd12b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/17bd12b2e866d3a0373bbbce4ccdaa3de1adb9b7))
- Use pytest template to improve tests and coverage reports (Adrien Berchet - [742eeed](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/742eeedf8933df96cc668be5e19e3108b1095c59))
- Update NeuroM dependency to 3.0.0 and update expected results (aleksei sanin - [2f8c447](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/2f8c447c2409c71570767fe4f1d62ab8dd70cf0e))
- Pin dependency versions in tests (Adrien Berchet - [593d67b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/593d67b3a64fb83735b471f23402b32eb0d04fc2))
- Fix vacuum example and use luigi-tools&gt;=0.0.15 (Alexis Arnaudon - [041efbe](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/041efbe2c2d1bf38abb9b328dd9d475165cff529))
- Some minor fix (Alexis Arnaudon - [5d242a1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/5d242a1368cd8de1af620f9b80a4efc706349d94))
- Use Kubernetes runner (Adrien Berchet - [941d190](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/941d190532cbe336c0416de632d9489fadef66c2))
- Drop support of py36 and py37 (Adrien Berchet - [62ed321](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/62ed32149f10e69df186a30671c51a2bd4a27f78))
- Use Gitlab registry in CI and remove several warnings (Adrien Berchet - [ab8a526](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/ab8a526d542f159fe22cb60e5fd3b32eed27ddac))
- cleanup vacuum example (Alexis Arnaudon - [6bacd19](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/6bacd190924cb9421d793ca529d21a6de6d0ee8e))
- Bumped NeuroTS and diameter-synthesis versions (Adrien Berchet - [872f8ec](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/872f8ecece01cfe404816efb703a8534880ddc53))
- Fix auto-release CI job (Adrien Berchet - [2e6eb0e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/2e6eb0ee8ca9367c05944eeaf4940b496b7a5277))

## [synthesis-workflow-v0.0.11](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.10...synthesis-workflow-v0.0.11)

> 25 May 2021

- Make choose_morphologies export scores (Adrien Berchet - [608b92f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/608b92f152fb94364a18758aaae3793134f661a7))
- Use luigi-tools&gt;=0.0.5 to automatically create parent directories of task targets (Adrien Berchet - [fa9beab](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/fa9beabde3b4417ac6742eca38dc9380c359f779))
- Create distributions for axon according to https://bbpcode.epfl.ch/code/\#/c/52107 (Adrien Berchet - [8fa1ecc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8fa1ecc61d556578ef5e7e76ff8d9e114b922a55))
- Update requirements (Adrien Berchet - [74a66b5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/74a66b501f1a3be579b881af29611625414cd863))
- density map tool (arnaudon - [8684580](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8684580ae52b2befda859daf2c1cdabba9cdd387))
- Update tox to for py36 only for linting (Adrien Berchet - [c44cb63](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c44cb636f1d4648ea26d90af1885efa4e3b084c8))
- Black the code with line length of 100 (Adrien Berchet - [f269215](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f26921556cae9336da843117b1779c8e5942f387))
- Fix compatibility with Py38 (Adrien Berchet - [cd07dec](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/cd07dec2146dbdc73cb3e263b4f08ce5ed78f1a4))
- Fix Py38 (Adrien Berchet - [3b93726](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/3b93726730961760069b4fbb3631a4208d849350))
- Use workflow rendering functions that were transferred to luigi-tools (Adrien Berchet - [fd582e5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/fd582e532c316ac33b56cbbfbcee2020e863c797))
- Use dir-diff-content, update to py38 and bump dependencies. (Adrien Berchet - [a9be8db](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a9be8db524ed2800c5a3726f78bf2d6d9ef78898))
- Merge "density map tool" (Alexis Arnaudon - [16af2e3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/16af2e3b73a94fa347979ae9ad547cbae7d05193))
- Import morph-validator functions, fix the tests and fix dependencies (Adrien Berchet - [c8d3fee](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c8d3fee829ef1ca10f9be45fbfa9d556cfb105d5))
- Use importlib in setup.py instead of imp (Adrien Berchet - [f6c43b3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f6c43b3f8c4410f1ebd32af4d986538c89cbacfd))
- Pin versions before moving to region-grower &gt;= 0.2 (Adrien Berchet - [a2529e1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a2529e16d9764d86de9a1cb6618f4b8194714a13))

## [synthesis-workflow-v0.0.10](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.9...synthesis-workflow-v0.0.10)

> 14 December 2020

- Update changelog (Adrien Berchet - [0a368c0](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/0a368c06b2961e79289a6daf26486358328f09d5))
- Fix PlotPathDistanceFits for mtypes with no fit (Adrien Berchet - [5341bae](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/5341baed94f36d8cb3e3e4bfb30cfd44d313d322))
- Fix BuildAxonMorphologies to use worker from placement_algorithm (Adrien Berchet - [5cc05de](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/5cc05de1c15bdded7fdd4d7eff5744fa383429d5))
- Use luigi-tools package (Adrien Berchet - [80d16ea](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/80d16ea00ec8b6e3e8ec67baded9f7de3e314618))
- Use luigi-tools==0.0.3 (Adrien Berchet - [19b4a66](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/19b4a66346b272d676bd0bfdc0c88459fe88b60e))
- Add methodology in the doc (Adrien Berchet - [63a8624](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/63a862490886b945f22d51377afbcb25241853f5))
- Improve doc: add link to TNS doc (Adrien Berchet - [e31dc56](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e31dc5677a995eec7b6e759d7099a2c3f453c80d))

## [synthesis-workflow-v0.0.9](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.8...synthesis-workflow-v0.0.9)

> 26 November 2020

- Update morph-tool requirements (Adrien Berchet - [1d4d46e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1d4d46e9d57620e9af87b7bd656944af4c86e6d5))
- Add a task to create annotation.json file (Adrien Berchet - [d983444](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d9834447a9c94b23a424df9685d204e3f455a2ad))
- Fix parallelization in vacuum synthesis (Adrien Berchet - [e2c68dc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e2c68dc3819d95feb5c253089be56c37c21cb9f7))
- Minor doc updates (Adrien Berchet - [1b2e5f1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1b2e5f1a25c50a443a36cc58b4f2f67d7ef9d966))

## [synthesis-workflow-v0.0.8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.7...synthesis-workflow-v0.0.8)

> 25 November 2020

- Updates the way the neuronDB files are found. (arnaudon - [4a4269e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/4a4269e930ec26ad4aec8c25f46bcfc1338c35ff))
- Add score matrix report (Adrien Berchet - [7d891ff](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/7d891ff113e8ff23103c3828e8ca4d6489725bca))
- Simplify doc and improve its generation (Adrien Berchet - [c73827d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c73827d3422fee6dd9365376547c24c1d8faa99e))

## [synthesis-workflow-v0.0.7](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.6...synthesis-workflow-v0.0.7)

> 18 November 2020

- added creation of thickness mask for Isocortex (arnaudon - [1ab0730](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1ab0730050e11907c1c1f0ff51175b1e1e135da1))
- Improve doc (Adrien Berchet - [a887fe8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a887fe8fe0bfac3a3feb9dc0e3291905665932be))
- Add examples of configuration files into the doc (Adrien Berchet - [6c6e94d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/6c6e94d1859dedd3a0e158e1063ab67448d5ba1e))

## [synthesis-workflow-v0.0.6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.5...synthesis-workflow-v0.0.6)

> 17 November 2020

- Improve doc and tests (Adrien Berchet - [017ba53](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/017ba53f652f1f735dd345fa62f7e1bffc44d2ff))

## [synthesis-workflow-v0.0.5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.4...synthesis-workflow-v0.0.5)

> 12 November 2020

- Update changelog (Adrien Berchet - [8950c50](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8950c50be68a24ba159ac2197dd2807b7a6ac9e1))

## [synthesis-workflow-v0.0.4](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.3...synthesis-workflow-v0.0.4)

> 12 November 2020

- Fix CLI for synthesis-workflow and MorphVal (Adrien Berchet - [a86eab5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a86eab58f541a19f0c826701010ec9f4aa2ce4d0))

## [synthesis-workflow-v0.0.3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.1...synthesis-workflow-v0.0.3)

> 11 November 2020

- Improve doc generation (Adrien Berchet - [66005e6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/66005e64b48ce535223347797187849a74e3e700))
- Fix requirements (Adrien Berchet - [7d1591d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/7d1591d28c8e4af72d072da2e9e1e9abd0fc94a4))
- Remove inheritance diagram from doc (Adrien Berchet - [c4c951e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c4c951e1f1e23e8f7a65b66d5e11e2036a7ec074))

## synthesis-workflow-v0.0.1

> 11 November 2020

- Initial commit (Adrien Berchet - [480db73](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/480db7332e5885cafe31821868d7fb94eee42d08))
- Fix logger and add logging of actual parameters after global variable processing (Adrien Berchet - [9984d60](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/9984d606d20692b94fba8d4632476361e9112614))
- Cleaning warnings and add a new one when Parameters are set to None (Adrien Berchet - [eb893e8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/eb893e83f896294667df0d728c57e724639dfe28))
- Use luigi's hook to log parameter values (Adrien Berchet - [d82615a](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d82615a6be3189fc618d95b5117938d04902df55))
- Set requirements (Adrien Berchet - [ed8c2b3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/ed8c2b33246bd286fa61df3fc8ca9b107787815e))
- Use joblib everywhere instead of multiprocessing (Adrien Berchet - [cc6c792](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/cc6c792215f32ecc73d87bf08bbf3c856ab0ca0d))
- Make plots more robust ; Fix collage tasks (Adrien Berchet - [a5fd59f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a5fd59f2d0a9fcb6423d2b7585756514a2853816))
- Initial empty repository (Dries Verachtert - [99d6ae2](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/99d6ae28f66ba3ca70bb957cc4b02265420b01aa))
- Setup pytest and update requirements (Adrien Berchet - [efe3ee1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/efe3ee1f0036895564e78ed27ac16ed77412f69c))
- Fix lint errors and add auto generation of version.py (Adrien Berchet - [1bb1a08](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1bb1a08b1f70b147b2d5ebb7ed2bdc9fa55c99f2))
- Reorganised the code ; Merged PlotMorphometrics and PlotVacuumMorphometrics tasks (Adrien Berchet - [d212244](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d212244160c912d46e4d7366000fe97be4b83475))
- Fix CI (Adrien Berchet - [1a8ac95](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1a8ac951c383cb1cc6835db93d72fcf3788725f0))
- small updates (arnaudon - [8156aaf](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8156aafd5e8655fc9f5ad2b7b02c33a96855d154))
- Fix axon choice and minor other cleanings (Adrien Berchet - [8af7b17](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8af7b17e5fb23b9cb4adb808f2f2f1502e6486c5))
- Fix axon_morphs_base_dir extraction (Adrien Berchet - [65b4574](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/65b4574b8dfb3522b58b07afb89c2af690f4225f))
- Fix create_axon_morphologies_tsv() (Adrien Berchet - [7813ece](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/7813ece00700950be6fddcd48b76a443151bed05))
- Hide some warnings (Adrien Berchet - [80b55a8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/80b55a8945b9b2495f290b170ed2f09765e9df23))
- collage update (arnaudon - [609fc39](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/609fc39ed4c4db84a86be085409ff2707292dc96))
- Update apical rescaling (Adrien Berchet - [25733bd](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/25733bd09c0c6cbbfae829d16320d2112f520441))
- Add CLI and rework logging (Adrien Berchet - [eaa27b9](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/eaa27b9a6430a6c1940ff8afeae9958f52889db1))
- several things in that commit: (arnaudon - [2b5a492](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/2b5a4927a0c93145157fe227a857561836416b50))
- Improve mechanism for global parameters (Adrien Berchet - [be274b3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/be274b30a96c245130c85d764250a7063daccff7))
- Fix logger for region-grower (Adrien Berchet - [a853dd5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a853dd56748530cf121b07241ee918a7c72038fe))
- Improve luigi tools (Adrien Berchet - [e8eae3b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e8eae3b9ec7b23ceb199cf8b0cf76fca750b385c))
- Add a test for luigi_tools.target_remove() (Adrien Berchet - [c6d1286](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c6d12869fb5a0b0506a93d2f5fbd9bf867dc9961))
- Transform all luigi.tasks into WorkflowTask and improve outputs (Adrien Berchet - [db706cc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/db706cc932e7151ba38ddf051c12b8184c58b929))
- Improve Synthesize dependency of PlotScale (Adrien Berchet - [10d930b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/10d930b224e32d3287137b09b5563a3adb9e0c72))
- Clean vacuum workflow (Adrien Berchet - [c7736c7](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c7736c75778d13179303bb0449d006d498d0fba6))
- Add task to checkout configuration from repository (Adrien Berchet - [b26d226](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/b26d226700864b26faf660800211378f94a0c878))
- Add diametrizer in vacuum synthesis (Adrien Berchet - [123c36a](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/123c36a9812466dd9cc165ac9a29765def6f43d4))
- Merge "Add task to checkout configuration from repository" (Adrien Berchet - [fcda863](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/fcda863439aaf3174c53d574d27dafd372298b6e))
- Add task to build a circuit MV3 file (Adrien Berchet - [4c19f80](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/4c19f80be14a31008d7a54d2daf80dc2864a3afa))
- Add task to build a morphs_df file (Adrien Berchet - [debf8a6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/debf8a6e4f96d3fc820595999c429267a29f2fa7))
- Add a new target class that add a prefix to its path (Adrien Berchet - [9be3c43](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/9be3c432b443c48ba55680b388739c92672e6536))
- generalise git clone task (arnaudon - [d6dc669](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d6dc6691ea5155c14ce1929ab3a4970a8a5b24ec))
- Add test for OutputLocalTarget with absolute path and prefix (Adrien Berchet - [c18c368](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c18c36895525ac2b8c30b7384b75f0e29fa964fe))
- Merge "Add test for OutputLocalTarget with absolute path and prefix" (Adrien Berchet - [69ffe7e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/69ffe7e4c4530cf92444bd407a1ed8c1c88c6f85))
- Improve validation configuration (Adrien Berchet - [75c7aa6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/75c7aa6d73ab88a11edf0cd7d7eadff401d12662))
- Minor fixes (Adrien Berchet - [31dddcb](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/31dddcb58e56866653ea0f7c11d894bf6eb042b8))
- fix bug in task shuffle and atlas get_layers (arnaudon - [f8060f6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f8060f68d4360551457b66ce559939f8237b25c4))
- Improve doc (Adrien Berchet - [73e5794](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/73e5794e0f3dccdef7a6e410c69bc6322aaf294f))
- Use specific targets to improve output directory tree (Adrien Berchet - [6ebdf5b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/6ebdf5be08cc52435d614eb404345a8a39d64cdf))
- Affine fit for path distance (arnaudon - [816e2ea](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/816e2ea066408d7d256e6c9565de5e9dd5ba7902))
- Add tests for the workflows and reorganize the code (Adrien Berchet - [f1608c9](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f1608c9711cad2e719042eea43c3ac80349b5461))
- Make the PlotCollage task much faster (Adrien Berchet - [cf27173](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/cf27173078957c6afb000cc9ffba4d9f05f99aff))
- Improve axon grafting (arnaudon - [f095bed](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f095bed32a7d01c29b02283da9317e02e868ed73))
- Merge "Make the PlotCollage task much faster" (Alexis Arnaudon - [95676a1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/95676a17ce52b24b77a92d072b2a57b2d532bba3))
- Improve parallel massively (arnaudon - [534cfdc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/534cfdc055a2bc389c6566ed3c7dbe47169cd41a))
- Add MorphVal library (Adrien Berchet - [3592568](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/3592568d8fbcbec2092129a1f0625532f710adea))
- Merge "Improve parallel massively" (Alexis Arnaudon - [1a1b8fa](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1a1b8fade68cbed928bafc0618f963d911ca0b86))
- Improved a few things (arnaudon - [3103154](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/31031549d907f8cad82466215c6c7e9aff9898ee))
- Fix morphval (Adrien Berchet - [bd722fe](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/bd722fef9591e44df162394c6db738f9def355cc))
- Optimize circuit slicing (Adrien Berchet - [8f2d7f2](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8f2d7f2c9b16e30688b643902fbf27ba4ff2eedd))
- Use absolute imports (Adrien Berchet - [7fc1c4d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/7fc1c4dd9311a4c76f6840126c91a31f3cb5b255))
- A None value for mtypes means that all mtypes are taken (Adrien Berchet - [8c92176](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8c92176968d75f52edf63119cb0842db02c398d3))
- Improve plot_collage (Adrien Berchet - [673282f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/673282f40dcd08bf94d57da7f9289ab29e7a2a90))
- Use 2 processes for tests (Adrien Berchet - [c4ddf19](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c4ddf199213c08862d0243b34ead38fdce69a429))
- Optimize validation.get_layer_info() (Adrien Berchet - [e47176b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e47176b535bae5a07ca77307299c109bc881fcb4))
- Optimize test distribution among processes and pylint (Adrien Berchet - [acd4962](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/acd4962a247cbca7dd639a5c961bc5322ccab277))
