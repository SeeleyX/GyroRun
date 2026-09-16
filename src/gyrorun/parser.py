# parser.py
import f90nml

def load_base_parameters(filepath):
    """Reads a GENE parameters file into a nested Python dictionary."""
    return f90nml.read(filepath)

def save_parameters(nml_dict, output_filepath):
    """Writes the dictionary back to a Fortran-compatible namelist."""
    nml_dict.write(output_filepath, force=True)