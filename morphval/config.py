import os

import yaml


EXAMPLE_CONFIG = '''\
default: &default
  stat_test: 'StatTests.ks'
  threshold: 0.1
  bins: 40
  criterion: 'dist'

BASAL_DATA: &basal_data
  section_lengths:
    stat_test: 'StatTests.ks'
    threshold: 0.1
    bins: 40
    criterion: 'dist'
  local_bifurcation_angles: *default
  partition: *default
  section_branch_orders: *default
  section_path_distances: *default
  section_radial_distances: *default
  number_of_sections_per_neurite: *default
  principal_direction_extents: *default
  total_length_per_neurite: *default
  total_length: *default
  number_of_neurites: *default
  number_of_bifurcations: *default
  section_tortuosity: *default

config:
    L1_HAC:
        basal_dendrite: *basal_data
'''


def load_config(config_path):
    assert os.path.exists(config_path), 'Missing config at: %s' % str(config_path)
    #  TODO: Should perform validation of config
    with open(config_path, 'r') as fd:
        return yaml.safe_load(fd)['config']
