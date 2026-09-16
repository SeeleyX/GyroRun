import pytest
import f90nml
from postprocess import parse_passive_species

TEST_CASES = [
    # Case 1: Standard unnumbered repeated &species blocks (omn, omt, uprim active)
    ("""
    &species
      name = 'main_ion', passive = F, omn = 2.0, omt = 2.0
    /
    &species
      name = 'trace_1', passive = T, omn = 1.0, omt = 0.0, uprim = 0.0
    /
    &species
      name = 'trace_2', passive = T, omn = 0.0, omt = 2.0, uprim = 1.5
    /
    """, ['omn', 'omt', 'uprim'], 2, [2, 3]),

    # Case 2: uprim zero/missing across all passive species -> drops uprim
    ("""
    &species
      name = 'trace_1', passive = T, omn = 2.0, omt = 0.0
    /
    &species
      name = 'trace_2', passive = T, omn = 0.0, omt = 3.0, uprim = 0.0
    /
    """, ['omn', 'omt'], 2, [1, 2]),

    # Case 3: No passive species in run (main ion + electron only) -> empty active_vars & species
    ("""
    &species
      name = 'ion', passive = F, omn = 2.2, omt = 2.2
    /
    &species
      name = 'electron', passive = F, omn = 2.2, omt = 2.2
    /
    """, [], 0, []),

    # Case 4: Realistic 6-species setup (2 active + 4 passive) with correct 1-based indexing [3, 4, 5, 6]
    ("""
    &species
      name = 'ion', passive = F, omn = 2.0, omt = 2.0
    /
    &species
      name = 'electron', passive = F, omn = 2.0, omt = 2.0
    /
    &species
      name = 'trace_1', passive = T, omn = 1.0, omt = 0.0, uprim = 0.0
    /
    &species
      name = 'trace_2', passive = T, omn = 0.0, omt = 1.0, uprim = 0.0
    /
    &species
      name = 'trace_3', passive = T, omn = 0.0, omt = 0.0, uprim = 1.0
    /
    &species
      name = 'trace_4', passive = T, omn = 1.0, omt = 1.0, uprim = 1.0
    /
    """, ['omn', 'omt', 'uprim'], 4, [3, 4, 5, 6]),

    # Case 5: Single passive species with omitted/default parameters
    ("""
    &species
      name = 'trace_only', passive = T, omn = 1.5
    /
    """, ['omn'], 1, [1]),

    # Case 6: Sub-threshold floating-point values (<= 1e-12) treated as zero/inactive
    ("""
    &species
      name = 'trace_1', passive = T, omn = 1.0e-13, omt = 2.0
    /
    """, ['omt'], 1, [1]),
]

@pytest.mark.parametrize("nml_str, expected_vars, expected_count, expected_indices", TEST_CASES)
def test_parse_passive_species(nml_str, expected_vars, expected_count, expected_indices):
    nml = f90nml.reads(nml_str)
    passive_species, active_vars = parse_passive_species(nml)
    
    assert len(passive_species) == expected_count
    assert active_vars == expected_vars
    assert [s['index'] for s in passive_species] == expected_indices