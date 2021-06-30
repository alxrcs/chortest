from numpy.core.fromnumeric import partition
import streamlit as sl
from tempfile import NamedTemporaryFile as tmpf
from PIL import Image
import sys
import os
from subprocess import run
from pathlib import Path
from glob import glob

# TODO: modify path so chorgram tools are reachable directly

GRAPHML_PATH = 'choreography.graphml'
OUTPNG_NAME = 'choreography.png'
OUTPUT_FOLDER = 'tmp'
DEFAULT_GCHOR = """A -> B : request;  { B -> A : ACK | B -> A : details} ; sel {A -> B: x + A -> B : y}"""
DEBUG = False

def call(cmd, debug=DEBUG, err_msg = None):
    res = run(cmd, shell=True, capture_output=True)
    if res.returncode != 0:
        if err_msg:
            sl.error(err_msg)
            sl.error(f'Output was: {str(res.stderr)}')
        else:
            sl.error(f'Error invoking {cmd}. Output was {str(res.stdout)}.')
        sl.stop()
    elif debug:
        sl.warning(f'Invoked "{cmd}". Output was {str(res.stdout)}')

def get_png_for_dot(dot_path):
    outpng_path = Path(dot_path).with_suffix('.png')
    call(f"dot -Tpng {dot_path} -o {outpng_path}")
    return Image.open(f'{outpng_path}','r')

def get_png_for_gc(gc_path):
    call(f"./chorgram/gc2dot -d {OUTPUT_FOLDER}/ {f.name}")
    dot_name = Path(f.name).with_suffix('.dot').name
    return get_png_for_dot(f'{OUTPUT_FOLDER}/{dot_name}')


# Cleanup tmp folder
call(f'rm {OUTPUT_FOLDER}/* -rf')

gchor = sl.text_area("G-chor source", value=DEFAULT_GCHOR)

if not gchor:
    sl.warning("Please write a g-chor")
    sl.stop()

f = tmpf("w")
f.write(gchor)
f.flush()


sl.image(get_png_for_gc(f.name), caption='G-choreography')

call(f'chortest project {f.name} --output-folder {OUTPUT_FOLDER}')

fsa_path = Path(f.name).with_suffix(".fsa")

call(f'chortest getdot {OUTPUT_FOLDER}/{fsa_path.name}')

participant_list = []

for machine_fname in glob(f'{OUTPUT_FOLDER}/machine*.dot'):
    sl.image(get_png_for_dot(machine_fname), caption=f'Projection for {machine_fname}')
    participant_name = Path(machine_fname.split('_')[-1]).stem
    participant_list.append(participant_name)

cut = sl.selectbox('CUT', participant_list)

call(f'chortest gentests --participant {cut} {OUTPUT_FOLDER}/{fsa_path.name}')

os.listdir(f'{OUTPUT_FOLDER}')

