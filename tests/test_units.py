"""Element library and unit conversion tests."""

import numpy as np

from modilab.units_io import element, element_source, element_symbols, to_real, unit_scales


def test_argon_time_unit_matches_source_table():
    # Schroeder (2015) Table I lists 1.95 ps for argon
    assert np.isclose(unit_scales("Ar")["tau_s"] * 1e12, 1.95, atol=0.01)


def test_temperature_conversion_uses_epsilon():
    for s in element_symbols():
        assert np.isclose(to_real("temperature", 1.0, s)[0], element(s)["epsilon_K"])


def test_every_element_has_required_fields_and_source():
    assert "doi" in element_source()
    for s in element_symbols():
        assert {"name_id", "name_en", "mass_u", "sigma_nm", "epsilon_K",
                "quantum_warning"} <= set(element(s))


def test_pressure_units_follow_epsilon_over_sigma_power():
    s = unit_scales("Ar")
    assert np.isclose(to_real("pressure3d", 1.0, "Ar")[0] * 1e6, s["epsilon_J"] / s["sigma_m"] ** 3)
    assert np.isclose(to_real("pressure2d", 1.0, "Ar")[0] / 1e3, s["epsilon_J"] / s["sigma_m"] ** 2)


def test_speed_distribution_stays_normalised_after_conversion():
    # f(v) dv is dimensionless, so f_real * v_real must equal f_reduced * v_reduced
    v_real = to_real("velocity", 1.0, "Kr")[0]
    f_real = to_real("speed_pdf", 1.0, "Kr")[0]
    assert np.isclose(v_real * f_real, 1.0)


def test_every_display_quantity_has_a_label_in_every_language():
    from modilab.i18n import LANGUAGES, load
    from modilab.ui.common import QUANTITIES
    for code in LANGUAGES:
        strings = load(code)
        for q in QUANTITIES:
            assert f"q.{q}" in strings, (code, q)
