import pytest
from glob import glob
from subprocess import CalledProcessError, call, check_output

def get_gc_filenames():
    return glob('tests/syntax/gc/*.gc')
    
@pytest.mark.skip
@pytest.mark.parametrize('gc_filename', get_gc_filenames())
def test_wf(gc_filename):
    try: 
        output_wf = str(check_output(['./wf', gc_filename]))
        assert ('ok' in output_wf) != ('fail' in gc_filename)
    except CalledProcessError as cpe:
        assert 'error' in gc_filename, "Unexpected parsing error"

@pytest.mark.parametrize('gc_filename', get_gc_filenames())
def test_wb(gc_filename):
    try: 
        output_wb = str(check_output(['./wb', gc_filename]))
        assert ('ok' in output_wb) != ('fail' in gc_filename)
    except CalledProcessError as cpe:
        assert 'error' in gc_filename, "Unexpected parsing error"

