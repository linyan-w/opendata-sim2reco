import pathlib
import pytest
import uproot

from sim2reco.io.branches import RECO_TREE_BRANCHES, TRUTH_TREE_BRANCHES

ROOT_FILE = pathlib.Path(__file__).resolve().parents[1] / "MasterAnaDev_mc_AnaTuple_run00113069_Playlist.root"
N_RECO = 20000
N_TRUTH = 40000


def _need_file():
    if not ROOT_FILE.exists():
        pytest.skip(f"example ntuple not present: {ROOT_FILE}")


@pytest.fixture(scope="session")
def reco():
    _need_file()
    return uproot.open(ROOT_FILE)["MasterAnaDev"].arrays(RECO_TREE_BRANCHES, entry_stop=N_RECO)


@pytest.fixture(scope="session")
def truth():
    _need_file()
    return uproot.open(ROOT_FILE)["Truth"].arrays(TRUTH_TREE_BRANCHES, entry_stop=N_TRUTH)


@pytest.fixture(scope="session")
def reco_typenames():
    _need_file()
    t = uproot.open(ROOT_FILE)["MasterAnaDev"]
    names = RECO_TREE_BRANCHES + [b + "_sz" for b in RECO_TREE_BRANCHES if b + "_sz" in t.keys()]
    return {k: t[k].typename for k in names}
