# Changelog

## [synthesis-workflow-v0.0.11](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.10...synthesis-workflow-v0.0.11)

> 25 May 2021

- Import morph-validator functions, fix the tests and fix dependencies (Adrien Berchet - [c8d3fee](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c8d3fee829ef1ca10f9be45fbfa9d556cfb105d5))
- Make choose_morphologies export scores (Adrien Berchet - [608b92f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/608b92f152fb94364a18758aaae3793134f661a7))
- Use dir-diff-content, update to py38 and bump dependencies. (Adrien Berchet - [a9be8db](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a9be8db524ed2800c5a3726f78bf2d6d9ef78898))
- Black the code with line length of 100 (Adrien Berchet - [f269215](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f26921556cae9336da843117b1779c8e5942f387))
- Pin versions before moving to region-grower &gt;= 0.2 (Adrien Berchet - [a2529e1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a2529e16d9764d86de9a1cb6618f4b8194714a13))
- Merge "density map tool" (Alexis Arnaudon - [16af2e3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/16af2e3b73a94fa347979ae9ad547cbae7d05193))
- Fix Py38 (Adrien Berchet - [3b93726](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/3b93726730961760069b4fbb3631a4208d849350))
- density map tool (arnaudon - [8684580](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8684580ae52b2befda859daf2c1cdabba9cdd387))
- Fix compatibility with Py38 (Adrien Berchet - [cd07dec](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/cd07dec2146dbdc73cb3e263b4f08ce5ed78f1a4))
- Use workflow rendering functions that were transfered to luigi-tools (Adrien Berchet - [fd582e5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/fd582e532c316ac33b56cbbfbcee2020e863c797))
- Use luigi-tools&gt;=0.0.5 to automatically create parent directories of task targets (Adrien Berchet - [fa9beab](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/fa9beabde3b4417ac6742eca38dc9380c359f779))
- Create distributions for axon according to https://bbpcode.epfl.ch/code/\#/c/52107 (Adrien Berchet - [8fa1ecc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8fa1ecc61d556578ef5e7e76ff8d9e114b922a55))
- Update requirements (Adrien Berchet - [74a66b5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/74a66b501f1a3be579b881af29611625414cd863))
- Use importlib in setup.py instead of imp (Adrien Berchet - [f6c43b3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f6c43b3f8c4410f1ebd32af4d986538c89cbacfd))
- Update tox to for py36 only for linting (Adrien Berchet - [c44cb63](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c44cb636f1d4648ea26d90af1885efa4e3b084c8))

## [synthesis-workflow-v0.0.10](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.9...synthesis-workflow-v0.0.10)

> 14 December 2020

- Use luigi-tools package (Adrien Berchet - [80d16ea](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/80d16ea00ec8b6e3e8ec67baded9f7de3e314618))
- Fix BuildAxonMorphologies to use worker from placement_algorithm (Adrien Berchet - [5cc05de](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/5cc05de1c15bdded7fdd4d7eff5744fa383429d5))
- Add methodology in the doc (Adrien Berchet - [63a8624](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/63a862490886b945f22d51377afbcb25241853f5))
- Update changelog (Adrien Berchet - [0a368c0](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/0a368c06b2961e79289a6daf26486358328f09d5))
- Use luigi-tools==0.0.3 (Adrien Berchet - [19b4a66](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/19b4a66346b272d676bd0bfdc0c88459fe88b60e))
- Fix PlotPathDistanceFits for mtypes with no fit (Adrien Berchet - [5341bae](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/5341baed94f36d8cb3e3e4bfb30cfd44d313d322))
- Improve doc: add link to TNS doc (Adrien Berchet - [e31dc56](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e31dc5677a995eec7b6e759d7099a2c3f453c80d))

## [synthesis-workflow-v0.0.9](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.8...synthesis-workflow-v0.0.9)

> 26 November 2020

- Add a task to create annotation.json file (Adrien Berchet - [d983444](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d9834447a9c94b23a424df9685d204e3f455a2ad))
- Fix parallelization in vacuum synthesis (Adrien Berchet - [e2c68dc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e2c68dc3819d95feb5c253089be56c37c21cb9f7))
- Minor doc updates (Adrien Berchet - [1b2e5f1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1b2e5f1a25c50a443a36cc58b4f2f67d7ef9d966))
- Update morph-tool requirements (Adrien Berchet - [1d4d46e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1d4d46e9d57620e9af87b7bd656944af4c86e6d5))

## [synthesis-workflow-v0.0.8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.7...synthesis-workflow-v0.0.8)

> 25 November 2020

- Simplify doc and improve its generation (Adrien Berchet - [c73827d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c73827d3422fee6dd9365376547c24c1d8faa99e))
- Add score matrix report (Adrien Berchet - [7d891ff](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/7d891ff113e8ff23103c3828e8ca4d6489725bca))
- Updates the way the neuronDB files are found. (arnaudon - [4a4269e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/4a4269e930ec26ad4aec8c25f46bcfc1338c35ff))

## [synthesis-workflow-v0.0.7](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/compare/synthesis-workflow-v0.0.6...synthesis-workflow-v0.0.7)

> 18 November 2020

- Improve doc (Adrien Berchet - [a887fe8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a887fe8fe0bfac3a3feb9dc0e3291905665932be))
- added creation of thickness mask for Isocortex (arnaudon - [1ab0730](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1ab0730050e11907c1c1f0ff51175b1e1e135da1))
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

- Optimize test distribution among processes and pylint (Adrien Berchet - [acd4962](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/acd4962a247cbca7dd639a5c961bc5322ccab277))
- Initial empty repository (Dries Verachtert - [99d6ae2](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/99d6ae28f66ba3ca70bb957cc4b02265420b01aa))
- Use 2 processes for tests (Adrien Berchet - [c4ddf19](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c4ddf199213c08862d0243b34ead38fdce69a429))
- Improve plot_collage (Adrien Berchet - [673282f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/673282f40dcd08bf94d57da7f9289ab29e7a2a90))
- A None value for mtypes means that all mtypes are taken (Adrien Berchet - [8c92176](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8c92176968d75f52edf63119cb0842db02c398d3))
- Use absolute imports (Adrien Berchet - [7fc1c4d](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/7fc1c4dd9311a4c76f6840126c91a31f3cb5b255))
- Add tests for the workflows and reorganize the code (Adrien Berchet - [f1608c9](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f1608c9711cad2e719042eea43c3ac80349b5461))
- Optimize circuit slicing (Adrien Berchet - [8f2d7f2](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8f2d7f2c9b16e30688b643902fbf27ba4ff2eedd))
- Fix morphval (Adrien Berchet - [bd722fe](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/bd722fef9591e44df162394c6db738f9def355cc))
- Add MorphVal library (Adrien Berchet - [3592568](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/3592568d8fbcbec2092129a1f0625532f710adea))
- Improved a few things (arnaudon - [3103154](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/31031549d907f8cad82466215c6c7e9aff9898ee))
- Merge "Improve parallel massively" (Alexis Arnaudon - [1a1b8fa](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1a1b8fade68cbed928bafc0618f963d911ca0b86))
- Improve parallel massively (arnaudon - [534cfdc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/534cfdc055a2bc389c6566ed3c7dbe47169cd41a))
- Improve axon grafting (arnaudon - [f095bed](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f095bed32a7d01c29b02283da9317e02e868ed73))
- Improve doc (Adrien Berchet - [73e5794](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/73e5794e0f3dccdef7a6e410c69bc6322aaf294f))
- Merge "Make the PlotCollage task much faster" (Alexis Arnaudon - [95676a1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/95676a17ce52b24b77a92d072b2a57b2d532bba3))
- Make the PlotCollage task much faster (Adrien Berchet - [cf27173](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/cf27173078957c6afb000cc9ffba4d9f05f99aff))
- Affine fit for path distance (arnaudon - [816e2ea](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/816e2ea066408d7d256e6c9565de5e9dd5ba7902))
-  Use specific targets to improve output directory tree (Adrien Berchet - [6ebdf5b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/6ebdf5be08cc52435d614eb404345a8a39d64cdf))
- fix bug in task shuffle and atlas get_layers (arnaudon - [f8060f6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/f8060f68d4360551457b66ce559939f8237b25c4))
- Minor fixes (Adrien Berchet - [31dddcb](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/31dddcb58e56866653ea0f7c11d894bf6eb042b8))
- Improve validation configuration (Adrien Berchet - [75c7aa6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/75c7aa6d73ab88a11edf0cd7d7eadff401d12662))
- Merge "Add test for OutputLocalTarget with absolute path and prefix" (Adrien Berchet - [69ffe7e](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/69ffe7e4c4530cf92444bd407a1ed8c1c88c6f85))
- Add test for OutputLocalTarget with absolute path and prefix (Adrien Berchet - [c18c368](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c18c36895525ac2b8c30b7384b75f0e29fa964fe))
- generalise git clone task (arnaudon - [d6dc669](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d6dc6691ea5155c14ce1929ab3a4970a8a5b24ec))
- Add a new target class that add a prefix to its path (Adrien Berchet - [9be3c43](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/9be3c432b443c48ba55680b388739c92672e6536))
- Add task to build a morphs_df file (Adrien Berchet - [debf8a6](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/debf8a6e4f96d3fc820595999c429267a29f2fa7))
- Add task to build a circuit MV3 file (Adrien Berchet - [4c19f80](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/4c19f80be14a31008d7a54d2daf80dc2864a3afa))
- Add diametrizer in vacuum synthesis (Adrien Berchet - [123c36a](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/123c36a9812466dd9cc165ac9a29765def6f43d4))
- Optimize validation.get_layer_info() (Adrien Berchet - [e47176b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e47176b535bae5a07ca77307299c109bc881fcb4))
- Clean vacuum workflow (Adrien Berchet - [c7736c7](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c7736c75778d13179303bb0449d006d498d0fba6))
- Add task to checkout configuration from repository (Adrien Berchet - [b26d226](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/b26d226700864b26faf660800211378f94a0c878))
- Improve Synthesize dependency of PlotScale (Adrien Berchet - [10d930b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/10d930b224e32d3287137b09b5563a3adb9e0c72))
- Transform all luigi.tasks into WorkflowTask and improve outputs (Adrien Berchet - [db706cc](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/db706cc932e7151ba38ddf051c12b8184c58b929))
- Add a test for luigi_tools.target_remove() (Adrien Berchet - [c6d1286](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/c6d12869fb5a0b0506a93d2f5fbd9bf867dc9961))
- Improve luigi tools (Adrien Berchet - [e8eae3b](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/e8eae3b9ec7b23ceb199cf8b0cf76fca750b385c))
- Fix logger for region-grower (Adrien Berchet - [a853dd5](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a853dd56748530cf121b07241ee918a7c72038fe))
- Improve mechanism for global parameters (Adrien Berchet - [be274b3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/be274b30a96c245130c85d764250a7063daccff7))
- several things in that commit: (arnaudon - [2b5a492](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/2b5a4927a0c93145157fe227a857561836416b50))
- Update apical rescaling (Adrien Berchet - [25733bd](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/25733bd09c0c6cbbfae829d16320d2112f520441))
- Add CLI and rework logging (Adrien Berchet - [eaa27b9](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/eaa27b9a6430a6c1940ff8afeae9958f52889db1))
- collage update (arnaudon - [609fc39](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/609fc39ed4c4db84a86be085409ff2707292dc96))
- Hide some warnings (Adrien Berchet - [80b55a8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/80b55a8945b9b2495f290b170ed2f09765e9df23))
- Fix create_axon_morphologies_tsv() (Adrien Berchet - [7813ece](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/7813ece00700950be6fddcd48b76a443151bed05))
- Fix axon_morphs_base_dir extraction (Adrien Berchet - [65b4574](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/65b4574b8dfb3522b58b07afb89c2af690f4225f))
- Fix axon choice and minor other cleanings (Adrien Berchet - [8af7b17](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8af7b17e5fb23b9cb4adb808f2f2f1502e6486c5))
- small updates (arnaudon - [8156aaf](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/8156aafd5e8655fc9f5ad2b7b02c33a96855d154))
- Fix CI (Adrien Berchet - [1a8ac95](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1a8ac951c383cb1cc6835db93d72fcf3788725f0))
- Reorganised the code ; Merged PlotMorphometrics and PlotVacuumMorphometrics tasks (Adrien Berchet - [d212244](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d212244160c912d46e4d7366000fe97be4b83475))
- Fix lint errors and add auto generation of version.py (Adrien Berchet - [1bb1a08](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/1bb1a08b1f70b147b2d5ebb7ed2bdc9fa55c99f2))
- Setup pytest and update requirements (Adrien Berchet - [efe3ee1](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/efe3ee1f0036895564e78ed27ac16ed77412f69c))
- Make plots more robust ; Fix collage tasks (Adrien Berchet - [a5fd59f](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/a5fd59f2d0a9fcb6423d2b7585756514a2853816))
- Use joblib everywhere instead of multiprocessing (Adrien Berchet - [cc6c792](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/cc6c792215f32ecc73d87bf08bbf3c856ab0ca0d))
- Set requirements (Adrien Berchet - [ed8c2b3](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/ed8c2b33246bd286fa61df3fc8ca9b107787815e))
- Use luigi's hook to log parameter values (Adrien Berchet - [d82615a](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/d82615a6be3189fc618d95b5117938d04902df55))
- Cleaning warnings and add a new one when Parameters are set to None (Adrien Berchet - [eb893e8](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/eb893e83f896294667df0d728c57e724639dfe28))
- Fix logger and add logging of actual parameters after global variable processing (Adrien Berchet - [9984d60](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/9984d606d20692b94fba8d4632476361e9112614))
- Initial commit (Adrien Berchet - [480db73](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/480db7332e5885cafe31821868d7fb94eee42d08))
- Merge "Add task to checkout configuration from repository" (Adrien Berchet - [fcda863](https://bbpgitlab.epfl.ch/neuromath/synthesis-workflow/commit/fcda863439aaf3174c53d574d27dafd372298b6e))
