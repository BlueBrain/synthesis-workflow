"""Workflow for neuronal synthesis validation."""
import pkg_resources
from morphio import SectionType

__version__ = pkg_resources.get_distribution("synthesis_workflow").version

STR_TO_TYPES = {
    "basal": SectionType.basal_dendrite,
    "apical": SectionType.apical_dendrite,
    "axon": SectionType.axon,
}
