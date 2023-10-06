from chortest.parsing import DOTBuilder, Parsers
import streamlit as st
from PIL import Image
import os
from subprocess import run
from pathlib import Path
from glob import glob
from lts import LTS


CHORGRAM_BASE_PATH = Path(__file__).parent.parent
GRAPHML_PATH = "choreography.graphml"
OUTPNG_NAME = "choreography.png"
OUTPUT_FOLDER = "tmp"
# DEFAULT_GCHOR = """A -> B : request;  { B -> A : ACK | B -> A : details} ; sel {A -> B: x + A -> B : y}"""
DEFAULT_GCHOR = "A -> B : req; sel {B -> A: ok + B -> A : err}; A->B : ack"
DEBUG = True

g_is_new_gc = False


def get_id_from(gc_str):
    return hex(abs(hash(gc_str)))[2:]


def call(cmd, err_msg=None, force_call=False, save_to=None):
    global g_is_new_gc
    if not g_is_new_gc and not force_call:
        return
    if DEBUG:
        st.spinner(f'Invoking "{cmd}"')

    if save_to:
        save_to = open(save_to, "w")
        res = run(cmd, shell=True, stdout=save_to)
        save_to.close()
    else:
        res = run(cmd, shell=True, capture_output=True)

    if res.returncode != 0:
        if err_msg:
            st.error(err_msg)
        st.error(f"{cmd}. \nOutput: {str(res.stdout)}\n Error: {str(res.stderr)}")
        st.stop()
    elif DEBUG:
        st.success(f"{cmd}")
        if res.stdout:
            st.markdown(f"```\n{res.stdout.decode('utf-8')}```")


def get_png_for_dot(dot_path, force_call=False):
    outpng_path = Path(dot_path).with_suffix(".png")

    call(f"dot -Tpng {dot_path} -o {outpng_path}", force_call=force_call)
    return Image.open(f"{outpng_path}", "r")


def get_png_for_gc(gc_path, gc_id):
    dot_path = Path(gc_path).with_suffix(".dot")
    call(f"gc2dot {gc_path}", save_to=dot_path)
    return get_png_for_dot(f"{OUTPUT_FOLDER}/{gc_id}/{dot_path.name}")


@st.cache(suppress_st_warning=True)
def generate_gchor(gc_text):
    global g_is_new_gc
    g_is_new_gc = True

    gc_id = get_id_from(gc_text)

    os.makedirs(f"{OUTPUT_FOLDER}/{gc_id}", exist_ok=True)

    with open(f"{OUTPUT_FOLDER}/{gc_id}/chor.gc", "w") as f:
        f.write(gc_text)

    img = get_png_for_gc(f.name, gc_id)

    return img, f.name, gc_id


gc_text = st.text_area("G-chor source", value=DEFAULT_GCHOR, height=20)

if not gc_text:
    st.warning("Please write a g-chor")
    st.stop()

bar = st.sidebar.progress(0)

# Cleanup tmp folder
# with st.spinner("Deleting old files..."):
# call(f"rm {OUTPUT_FOLDER}/* -rf", is_new_gc)
# bar.progress(10)

img, fname, gc_id = generate_gchor(gc_text)

st.image(img, caption="G-choreography")

with st.spinner("Projecting..."):
    call(
        f"chortest project {fname} --output-folder {OUTPUT_FOLDER}/{gc_id}",
    )
    bar.progress(30)

fsa_path = Path(fname).with_suffix(".fsa")

with st.spinner("Getting dot files for projections..."):
    call(
        f"chortest getdot {OUTPUT_FOLDER}/{gc_id}/{fsa_path.name}",
    )
    bar.progress(50)

participant_list = []

for machine_fname in glob(f"{OUTPUT_FOLDER}/{gc_id}/machine*.dot"):
    st.image(get_png_for_dot(machine_fname), caption=f"Projection for {machine_fname}")
    participant_name = Path(machine_fname.split("_")[-1]).stem
    participant_list.append(participant_name)

cut = st.sidebar.selectbox("CUT", participant_list)

with st.spinner(f"Generating tests for participant {cut}..."):
    call(
        f"chortest gentests --participant {cut} {OUTPUT_FOLDER}/{gc_id}/{fsa_path.name}",
        force_call=True,
    )
    bar.progress(70)

tests_dir = f"{OUTPUT_FOLDER}/{gc_id}/{fsa_path.stem}_tests/{cut}/"
total_tests = len(os.listdir(tests_dir))
st.sidebar.write(f"Number of tests generated: {total_tests}")

test_no = st.sidebar.number_input(
    "Select a test", min_value=0, max_value=total_tests - 1
)

if not st.sidebar.button("Generate LTS"):
    bar.progress(100)
    st.stop()

test_path = f"{tests_dir}/test_{test_no}"

st.write(f"chortest genlts {test_path}/test_{test_no}.fsa")
with st.spinner(f"Generating LTS for test {test_no}"):
    call(
        f"chortest genlts {test_path}/test_{test_no}.fsa", force_call=True
    )  # TODO: Add the CUT substitution here with --cut-filename
    bar.progress(80)

lts_path = f"{test_path}/test_{test_no}_ts5.dot"

with st.spinner(f"Generating png for dot file {lts_path}..."):
    st.image(get_png_for_dot(lts_path, force_call=True), caption=f"LTS for {lts_path}")

with st.spinner(f"Parsing LTS..."):
    lts = Parsers.parseFile(str(lts_path))
    st.sidebar.write("Number of nodes:", len(lts.configurations))
    st.sidebar.write("Number of transitions:", len(lts.transitions))
    bar.progress(100)
