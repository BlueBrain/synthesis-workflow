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

.PHONY: help Makefile clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	# Generate images
	mkdir -p source/autoapi/tasks/validation/ source/autoapi/tasks/vacuum_synthesis/
	pdftocairo -png -r 300 -f 5 -l 5 ../tests/data/in_small_O1/out/validation/scales/statistics.pdf source/autoapi/tasks/validation/scale_statistics
	# pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_small_O1/out/validation/collages/L3_TPC:A.pdf source/autoapi/tasks/validation/collages
	pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_small_O1/out/validation/path_distance_fit.pdf source/autoapi/tasks/validation/path_distance_fit
	pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_vacuum/out/validation/morphometrics/morphometrics_basal_dendrite.pdf source/autoapi/tasks/validation/morphometrics
	# pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_vacuum/out/validation/vacuum_morphologies.pdf source/autoapi/tasks/vacuum_synthesis/vacuum_morphologies
	pdftocairo -png -r 300 -f 1 -l 1 ../tests/data/in_small_O1/out/validation/score_matrix_reports.pdf source/autoapi/tasks/validation/score_matrix_reports

	# Generate dependency graphs
	cd ../tests/data/in_small_O1; synthesis_workflow -dg ../../../docs/source/autoapi/tasks/workflows/ValidateSynthesis.dot ValidateSynthesis; cd ../../../docs
	cd ../tests/data/in_small_O1; synthesis_workflow -dg ../../../docs/source/autoapi/tasks/workflows/ValidateRescaling.dot ValidateRescaling; cd ../../../docs
	cd ../tests/data/in_vacuum; synthesis_workflow -dg ../../../docs/source/autoapi/tasks/workflows/ValidateVacuumSynthesis.dot ValidateVacuumSynthesis; cd ../../../docs

	# Generate the documentation
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm -rf $(BUILDDIR) $(SOURCEDIR)/examples $(SOURCEDIR)/generated
