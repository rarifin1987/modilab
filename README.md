# MoDiLab

**MoDiLab** is a bilingual (Indonesian and English) virtual laboratory for
teaching molecular dynamics in computational materials science courses.
Students run two dimensional Lennard-Jones simulations through a browser
based graphical interface, without writing code, and explore how
microscopic motion gives rise to structure, transport and thermodynamics.

*Versi bahasa Indonesia ada di bagian akhir dokumen ini.*

## Features (v0.2.0)

* Graphical interface in Indonesian or English, switchable at any time
  without losing results; numbers follow the decimal convention of the
  selected language
* Gas, liquid and solid presets, plus custom settings
* NVE ensemble, Berendsen thermostat and velocity rescaling
* Gradual heating or cooling
* Animated atoms coloured by speed
* Kinetic, potential and total energy, temperature and pressure
* Radial distribution function g(r)
* Mean square displacement and diffusion coefficient (Einstein relation)
* Speed distribution compared with the 2D Maxwell-Boltzmann distribution
* Conversion of reduced units to argon
* Built in theory summary and five guided experiments
* Export to CSV and to extended XYZ trajectories readable by OVITO and ASE

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application opens in the browser at http://localhost:8501.

For classroom use without local installation, the repository can be
deployed on Streamlit Community Cloud; students then only need the link.

## Tests

```bash
python -m pytest tests
```

The test suite checks that forces equal the negative energy gradient,
that total energy is conserved in NVE, that total momentum stays zero,
that the thermostat reaches its target, that unstable runs are flagged,
and that every translation file has the same keys and placeholders as
the English reference.

## Code layout

```
app.py                  graphical interface (Streamlit)
modilab/core.py         potential, forces, integrator, thermostats
modilab/analysis.py     g(r), MSD, diffusion, speed distribution
modilab/units_io.py     argon units and data export
modilab/i18n.py         translation layer
modilab/locales/*.json  user facing text, one file per language
tests/                  automated tests
```

## Adding a language

1. Copy `modilab/locales/en.json` to `modilab/locales/<code>.json`
2. Translate the values, keeping every key and every `{placeholder}`
3. Register the language in `LANGUAGES` in `modilab/i18n.py`
4. Run `python -m pytest tests` to confirm the file is consistent

## Known limitations

* Force evaluation scales as O(N²), practical up to about 400 particles
* The initial configuration is a square lattice, whereas the stable 2D
  Lennard-Jones solid is hexagonal
* Two dimensions only

## Licence

MoDiLab is released under the MIT License. See the `LICENSE` file for details.

---

## Ringkasan dalam bahasa Indonesia

MoDiLab adalah laboratorium virtual dinamika molekuler dwibahasa untuk
mata kuliah Komputasi Material. Bahasa antarmuka dapat dipilih di bagian
atas panel kiri; bahasa Indonesia adalah bawaan.

Cara menjalankan:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Seluruh teks antarmuka tersimpan di `modilab/locales/id.json` dan
`modilab/locales/en.json`. Untuk mengubah kalimat di aplikasi, cukup
sunting berkas tersebut tanpa menyentuh kode program.
