import json

from backend.provenance import citation_records, software_provenance
from tests.test_frozen_scba import make_system


def test_software_provenance_is_json_serializable_and_reports_dependencies():
    provenance = software_provenance()
    assert provenance["software"]["name"] == "NEGF-TDPSCBA"
    assert "commit" in provenance["software"]["git"]
    assert provenance["runtime"]["dependencies"]["numpy"] is not None
    json.dumps(provenance)


def test_system_provenance_and_citations_report_selected_method():
    system = make_system(pulse_protocol="upward", scba_mode="weak_born")
    provenance = system.provenance()
    assert provenance["calculation"]["pulse_protocol"] == "upward"
    assert provenance["calculation"]["stationary_approximation"] == "weak_born"
    assert provenance["calculation"]["units"]["backend_energy"] == "Gamma"
    assert any(
        record.get("doi") == "10.1103/PhysRevB.74.085324"
        for record in system.citations()
    )
    assert citation_records() == system.citations()
    json.dumps(provenance)
