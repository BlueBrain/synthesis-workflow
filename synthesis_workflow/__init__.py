"""Workflow for neuronal synthesis validation."""
from morphio import SectionType


STR_TO_TYPES = {
    "basal": SectionType.basal_dendrite,
    "apical": SectionType.apical_dendrite,
    "axon": SectionType.axon,
}
