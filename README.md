# MoDiLab

**MoDiLab** is a bilingual (Indonesian and English) virtual laboratory for
teaching molecular dynamics in computational materials science courses.
It combines structured learning modules, each with its own guided
simulation, analysis questions and quiz, with an open ended simulator.
Students work in the browser without writing code.

*Ringkasan bahasa Indonesia ada di bagian akhir dokumen ini.*

## Learning modules (v0.3.1)

| Module | Topic | Guided simulation |
|---|---|---|
| 1 | Interatomic forces | Interactive U(r) and F(r) for noble gases, real vs reduced units |
| 2 | Equations of motion | Two atom dimer: explicit Euler vs velocity Verlet, phase space |
| 3 | Periodic box and units | Periodic images animation, unit conversion calculator |
| 4 | Temperature and ensembles | NVE vs NVT (Berendsen) from identical initial states |
| 5 | Maxwell-Boltzmann distribution | Relaxation from equal speeds, in 2D or 3D |
| 6 | Structure of matter | g(r) and coordination number for gas, liquid and solid; 2D hexagonal and 3D fcc |

Each module contains learning objectives, concept text with equations,
the simulation, analysis questions and a three question quiz. A free
laboratory offers the full 2D simulator with energy, g(r), MSD, speed
distribution and data export.

Planned: equation of state, corresponding states, phase transitions and
transport (v0.4); binary mixtures and the limits of the Lennard-Jones
model (v0.5).

## Display units

A global setting in the sidebar switches every plot and value between
reduced units (σ, ε, m) and real units (K, ps, Å, meV, m/s, MPa or mN/m,
g/cm³ or nm⁻², cm²/s) for a chosen reference element. Simulation inputs
stay in reduced units, with their real equivalents shown beneath them.
In 2D, density is per unit area and pressure is force per unit length, so
these values are for comparison only.

## Element library

Noble gas parameters (He, Ne, Ar, Kr, Xe) are taken from
D. V. Schroeder, *Interactive molecular dynamics*, Am. J. Phys. 83, 210
(2015), https://doi.org/10.1119/1.4901185, Table I, where they are adapted
from Maitland et al., *Intermolecular Forces* (1981). As noted in that
source, ε values can vary by 10 % or more depending on the fitted data.
He and Ne are flagged in the interface because quantum effects make
classical results unrealistic at low temperature.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m pytest tests
```

The 43 tests cover force and energy consistency in 2D and 3D, energy
conservation, momentum conservation, thermostat behaviour, lattice
geometry, relaxation to the Maxwell-Boltzmann distribution, the
second order accuracy of velocity Verlet, unit conversion against the
source table and dimensional consistency, consistency of all translation and content files, and a
start up test of the full application in both languages.

## Code layout

```
app.py                     multipage entry point
modilab/core.py            engine: lattices, forces, integrator, thermostats, dimer model
modilab/analysis.py        g(r), coordination number, MSD, diffusion, speed distribution
modilab/units_io.py        element library, unit conversion, data export
modilab/data/elements.json element parameters with source
modilab/i18n.py            translation layer and content loader
modilab/locales/*.json     interface text, one file per language
modilab/content/<lang>/    module text (Markdown) and quizzes (JSON)
modilab/ui/                pages, shared components, free laboratory
tests/                     automated tests
```

## Editing the learning material

Module text lives in `modilab/content/<lang>/mXX.md`. The line
`<!-- SIM -->` marks where the simulation appears; text before it is the
introduction and text after it is shown below the simulation. Quizzes are
in `modilab/content/<lang>/quiz.json`. Run the tests after editing to make
sure both languages stay consistent.

## Adding a language

Copy `modilab/locales/en.json` and the folder `modilab/content/en/` to the
new language code, translate the values and texts, register the language
in `LANGUAGES` in `modilab/i18n.py`, and run the tests.

## Known limitations

* Force evaluation scales as O(N²); 3D modules are limited to 256 atoms
* Berendsen thermostat only (no Nosé-Hoover or Langevin yet)
* Lennard-Jones parameters are provided for noble gases only

## Licence

MoDiLab is released under the MIT License. See the `LICENSE` file.

---

## Ringkasan dalam bahasa Indonesia

MoDiLab versi 0.3.1 berisi enam modul pembelajaran berurutan (gaya
antaratom, persamaan gerak, kotak periodik dan satuan, suhu dan ensemble,
distribusi Maxwell-Boltzmann, struktur zat) dan satu laboratorium bebas.
Setiap modul memuat tujuan pembelajaran, uraian konsep, simulasi terpandu,
pertanyaan analisis, dan kuis. Bahasa dipilih di panel kiri.

Satuan tampilan (tereduksi atau nyata) dan elemen acuan dipilih di panel
kiri dan berlaku untuk seluruh modul serta laboratorium.

Materi setiap modul dapat disunting di `modilab/content/id/` (bahasa
Indonesia) dan `modilab/content/en/` (bahasa Inggris) tanpa mengubah kode.
Jalankan `python -m pytest tests` setelah menyunting untuk memastikan kedua
bahasa tetap konsisten.
