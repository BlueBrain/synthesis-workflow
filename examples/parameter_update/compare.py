"""Example of comparison."""

import json

import pandas as pd
from neurots.utils import convert_from_legacy_neurite_type

from synthesis_workflow.utils import apply_parameter_diff
from synthesis_workflow.utils import create_parameter_diff

if __name__ == "__main__":
    param_spec = convert_from_legacy_neurite_type(
        json.load(open("out/synthesis_input/tmd_specific_parameters.json"))
    )
    param = json.load(open("out/synthesis/neurots_input/tmd_parameters.json"))
    custom_values = create_parameter_diff(param, param_spec)
    custom_values.to_csv("custom_parameters.csv", index=False)

    custom_values = pd.read_csv("custom_parameters.csv")
    apply_parameter_diff(param, custom_values)
    json.dump(param, open("updated_tmd_parameters.json", "w"), indent=4)
