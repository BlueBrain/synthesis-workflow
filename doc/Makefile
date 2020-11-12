# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# Generate images
	mkdir -p source/autoapi/tasks/validation/ source/autoapi/tasks/vacuum_synthesis/
	pdftocairo -png -r 300 -f 5 -l 5 ../tests/data/in_small_O1/out/validation/scales/statistics.pdf source/autoapi/tasks/validation/scale_statistics
	pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_small_O1/out/validation/collages/L3_TPC:A.pdf source/autoapi/tasks/validation/collages
	pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_small_O1/out/validation/path_distance_fit.pdf source/autoapi/tasks/validation/path_distance_fit
	pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_vacuum/out/validation/morphometrics/morphometrics_basal_dendrite.pdf source/autoapi/tasks/validation/morphometrics
	pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_vacuum/out/validation/vacuum_morphologies.pdf source/autoapi/tasks/vacuum_synthesis/vacuum_morphologies

	# Generate dependency graphs
	cd ../tests/data/in_small_O1; synthesis_workflow -dg ../../../doc/source/autoapi/tasks/workflows/ValidateSynthesis.dot ValidateSynthesis; cd ../../../doc
	cd ../tests/data/in_small_O1; synthesis_workflow -dg ../../../doc/source/autoapi/tasks/workflows/ValidateRescaling.dot ValidateRescaling; cd ../../../doc
	cd ../tests/data/in_vacuum; synthesis_workflow -dg ../../../doc/source/autoapi/tasks/workflows/ValidateVacuumSynthesis.dot ValidateVacuumSynthesis; cd ../../../doc

	# Generate the documentation
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
