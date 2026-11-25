import pytest
import imaging.cmyk_black as mod

CASES = [
    ("tic_constraint", (1,)),
    ("region_gcr", (1,)),
    ("sync_sharpen", (1,)),
    ("gray_balance_rebuild", (1,)),
    ("paper_white_adapt", (1,)),
    ("page_order_bind", (1,)),
    ("load_rgb", (1,)),
    ("convert_cmyk", (1,)),
    ("proof_preview", (1,)),
    ("present_print", (1,)),
    ("validate_print", (1,)),
    ("metrics_print", (1,)),
    ("audit_print", (1,)),
    ("fallback_print", (1,)),
    ("determinism_print", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
