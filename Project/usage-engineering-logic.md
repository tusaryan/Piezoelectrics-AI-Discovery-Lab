# Piezoelectric Material Usage Prediction Logic

## Reference Document for ML App Enhancement (Claude Opus Prompt)

---

## CONTEXT FOR AGENT

You are enhancing a machine learning web application that predicts properties of piezoelectric materials — specifically **d33 (pC/N)**, **Tc (°C)**, and **Hardness (HV)** — from bulk ceramic or composite formulas. The model already predicts these values. The task now is to **add a downstream "Usage Prediction" module** that takes any subset of {d33, Tc, Hardness} — whichever the user has predicted or entered — and recommends suitable real-world applications for that material.

The logic below is grounded in peer-reviewed literature, USPTO patents, and industry sources. Implement it as a **rule-based scoring engine with confidence tiers**, optionally backed by the trained ML model where available.

---

## PART 1 — PROPERTY RANGES & SCIENTIFIC MEANING

### 1.1 — d33 (Piezoelectric Charge Coefficient, pC/N)

This is the most directly useful property for usage prediction. It measures how much electrical charge is generated per unit of applied force (or vice versa for actuators).

| Range (pC/N) | Tier Label | Meaning                                                                  |
| ------------ | ---------- | ------------------------------------------------------------------------ |
| < 10         | Ultra-Low  | Non-ferroelectric or very weak piezo; quartz-class                       |
| 10 – 50      | Low        | Moderate sensing only; high-temp specialized materials                   |
| 50 – 200     | Moderate   | Standard sensing and transducer range                                    |
| 200 – 500    | High       | Actuator-grade; medical ultrasound, energy harvesting                    |
| 500 – 1000   | Very High  | High-power actuation, precision positioning                              |
| > 1000       | Ultra-High | Single-crystal class (PMN-PT, PIN-PMN-PT); medical imaging, implantables |

**References:**

- Relaxor-PT single crystals (PMN-PT, PIN-PMN-PT): d33 = 1500–2700+ pC/N — used in medical ultrasound transducers (PMC4223717)
- Sm-PMN-PT: d33 up to 4000 pC/N — deep brain stimulation energy harvesters (Science Advances, PMC9012468)
- Perovskite polycrystalline ceramics (PZT): d33 = 100–1000 pC/N — widest application range
- Tungsten Bronze / BLSF (Aurivillius) structures: d33 = 10–100 pC/N — high-temperature niche sensors
- Quartz / Langasites: d33 < 10 pC/N — ultra-stable but low sensitivity (frequency references, pressure gauges)

**Key note:** d33 alone is not sufficient for actuators. In power ultrasonics, the mechanical quality factor Qm matters more — **high d33 + low Qm = heat generation risk** (soft PZT problem in high-duty-cycle drives).

---

### 1.2 — Tc (Curie Temperature, °C)

This is the hard upper limit of piezoelectric function. Material must operate at **T < Tc/2** for stable long-term use (practical depoling safety margin). The maximum operating temperature is typically **0.5 × Tc**.

| Tc Range (°C) | Practical Max Op. Temp | Tier Label                                   |
| ------------- | ---------------------- | -------------------------------------------- |
| < 100 °C      | < 50 °C                | Very Low — lab/cryogenic only                |
| 100 – 200 °C  | 50 – 100 °C            | Low — room temp / consumer electronics       |
| 200 – 350 °C  | 100 – 175 °C           | Medium — standard industrial                 |
| 350 – 500 °C  | 175 – 250 °C           | High — automotive, aerospace                 |
| 500 – 700 °C  | 250 – 350 °C           | Very High — power plant, jet engine adjacent |
| > 700 °C      | > 350 °C               | Extreme — nuclear, deep well, turbine        |

**References:**

- PMN-PT single crystals: Tc ~ 70–120°C → max useful use ~90°C (high d33 but thermally limited)
- PZT Navy Type II: Tc ~ 350°C → stable to ~200°C (USPTO 8518291)
- Bismuth titanate (Bi4Ti3O12): Tc ~ 650°C → sensors to ~450°C (USPTO 7658111)
- PLS ferroelectrics (Nd2Ti2O7): Tc > 1000°C, depoling stable to ~1400°C — nuclear/turbine sensors (PMC4823784)
- Standard safe operating rule: use material below Tc/2 for stable long-term polarization

---

### 1.3 — Hardness (Vickers HV)

Hardness in piezoceramics determines **mechanical durability, wear resistance, and structural load-bearing capacity**. It indirectly relates to application environment severity.

| HV Range      | Tier Label   | Meaning                                                      |
| ------------- | ------------ | ------------------------------------------------------------ |
| < 300 HV      | Soft/Fragile | Polymer piezo, foam, PVDF-type; flexible wearables           |
| 300 – 600 HV  | Moderate     | Standard soft PZT ceramics (PZT-5 series); sensitive sensors |
| 600 – 900 HV  | Medium-Hard  | Hard PZT ceramics (PZT-4, PZT-8); industrial actuation       |
| 900 – 1200 HV | Hard         | High-density sintered ceramics; heavy-duty transducers       |
| > 1200 HV     | Very Hard    | Structural-grade piezoceramics; harsh environment / military |

**Notes:**

- Hard PZT (acceptor-doped): higher Qm, lower dielectric loss, better for high-power ultrasonics but lower d33 (ScienceDirect, high-power piezo review)
- Soft PZT (donor-doped): higher d33 and permittivity but lower HV and thermal stability
- PZT ceramics with Vickers hardness ~9.7 GPa (~990 HV) show compressive strength ~500 MPa and are viable for high-power device applications (ScienceDirect)
- Fracture toughness of piezoceramics: 0.5–2.0 MPa·m^0.5; low HV materials are more susceptible to impact failure

---

## PART 2 — INDIVIDUAL PROPERTY → USE CASE MAPPING (Single-Feature Logic)

Use these rules when only **one** property is available.

### 2.1 — When ONLY d33 is known

```
IF d33 < 10 pC/N:
  → Frequency reference oscillators, quartz-type pressure gauges,
    high-precision laboratory sensors, optical modulators

IF 10 ≤ d33 < 50 pC/N:
  → High-temperature gas sensors (aerospace/nuclear adjacent),
    harsh environment sensing (limited actuation),
    acoustic emission monitoring

IF 50 ≤ d33 < 200 pC/N:
  → Flow meters, accelerometers, NDT (non-destructive testing) sensors,
    impact sensors, sonar receivers (hydrophones),
    industrial vibration monitoring

IF 200 ≤ d33 < 500 pC/N:
  → Standard piezoelectric actuators, ink-jet print heads,
    energy harvesting (ambient vibration),
    medical ultrasound transducers (mid-range),
    fuel injectors, precision positioning stages

IF 500 ≤ d33 < 1000 pC/N:
  → High-power ultrasonic cleaning/welding,
    precision nanopositioning actuators,
    high-sensitivity medical imaging,
    piezoelectric motors (USM),
    sonar projectors (active transmit)

IF d33 ≥ 1000 pC/N:
  → Medical phased-array imaging transducers,
    intravascular ultrasound (IVUS),
    implantable energy harvesters (e.g., DBS devices),
    photoacoustic imaging,
    high-resolution underwater imaging sonar,
    flexible/wearable biomedical sensors
```

---

### 2.2 — When ONLY Tc is known

```
IF Tc < 100 °C:
  → Lab instruments (controlled temperature environments only),
    cryogenic sensing, NOT suitable for any field deployment

IF 100 ≤ Tc < 200 °C:
  → Consumer electronics (wearables, smartphones, IoT sensors),
    room-temperature energy harvesters,
    low-power biomedical implants (body temp: ~37°C → safe margin)

IF 200 ≤ Tc < 350 °C:
  → Industrial automation sensors, HVAC systems,
    flow measurement, accelerometers for machinery,
    standard medical devices, precision actuators (lab/factory)

IF 350 ≤ Tc < 500 °C:
  → Automotive sensors (engine bay proximity, exhaust monitoring),
    aerospace structural health monitoring,
    geothermal exploration sensors,
    high-temp NDT transducers

IF 500 ≤ Tc < 700 °C:
  → Aero-engine monitoring (turbine adjacent),
    industrial furnace/reactor sensing,
    military platform sensors (harsh environment),
    down-hole oil & gas sensors,
    nuclear plant condition monitoring (peripheral)

IF Tc ≥ 700 °C:
  → Nuclear reactor core adjacent sensors,
    deep-earth drilling / geothermal (>300°C bore environments),
    jet turbine blade monitoring,
    extreme-temperature industrial process control
```

---

### 2.3 — When ONLY Hardness (HV) is known

```
IF HV < 300:
  → Flexible/stretchable piezoelectrics (PVDF, polymer),
    wearable health monitors (skin-mounted sensors),
    e-skin / tactile sensing,
    soft robotics pressure sensing

IF 300 ≤ HV < 600:
  → Standard sensor applications (soft PZT grade),
    medical ultrasound probes (sensitivity-focused),
    hydrophones (low-power listening mode),
    piezoelectric microphones,
    low-power energy harvesting

IF 600 ≤ HV < 900:
  → Industrial transducers (hard PZT grade),
    ultrasonic cleaning/welding (moderate duty),
    flow meters, level sensors,
    automotive knock sensors,
    precision industrial actuators

IF 900 ≤ HV < 1200:
  → High-load industrial transducers,
    heavy-duty sonar transducers,
    high-power ultrasonic processing,
    structural health monitoring in harsh environments

IF HV ≥ 1200:
  → Military sonar (shock-hardened requirements),
    underwater explosion sensing,
    defense/aerospace structural monitoring,
    extreme mechanical load environments,
    high-pressure industrial processing (underwater, deep sea)
```

---

## PART 3 — COMBINED PROPERTY → USE CASE LOGIC (Multi-Feature Rules)

These are more accurate than single-feature predictions. Use priority-weighted scoring.

### 3.1 — d33 + Tc Combined (Most Powerful Combination)

```
[HIGH d33 + LOW Tc] (e.g., d33 > 500, Tc < 200°C):
  → Best fit: Medical ultrasound imaging, implantable devices,
    wearable/IoT sensors, lab biomedical instruments
  → Avoid: Any application near heat sources or outdoor field deployment
  → Example material: PMN-PT single crystals

[HIGH d33 + MEDIUM Tc] (e.g., d33 200–500, Tc 200–350°C):
  → Best fit: Standard actuators, energy harvesters, industrial sensors,
    precision positioning, ink-jet heads, fuel injection
  → Example material: PZT-5A (Tc ~350°C, d33 ~374 pC/N)

[HIGH d33 + HIGH Tc] (e.g., d33 > 200, Tc > 400°C):
  → IDEAL COMBINATION — rare but highly sought
  → Best fit: Aerospace sensors, high-temperature actuators,
    automotive powertrain, high-temp NDT
  → Example: KNN-LiNbO3 TGG ceramics (d33=280 pC/N, Tc=430°C)
  → Note: ACS paper (2016) explicitly targets this combo for industrial demand

[LOW d33 + HIGH Tc] (e.g., d33 < 50, Tc > 500°C):
  → Best fit: Harsh-environment sensing (not actuation),
    high-temp vibration/acoustic emission sensing,
    nuclear/aerospace monitoring where sensitivity is secondary to stability
  → Example: Langasites, BLSF ceramics, modified PbTiO3

[LOW d33 + LOW Tc]:
  → Very limited use; candidate for quartz-class reference applications only
```

---

### 3.2 — d33 + Hardness Combined

```
[HIGH d33 + LOW HV] (soft material, high sensitivity):
  → Flexible sensors, wearable bioelectronics, polymer composites,
    stretchable energy harvesters, skin-conformable devices

[HIGH d33 + MEDIUM HV] (soft PZT territory):
  → Medical ultrasound probes, low-power hydrophones,
    precision actuators, energy harvesting from ambient sources

[HIGH d33 + HIGH HV] (hard PZT or dense ceramic):
  → High-power ultrasonics (welding, cleaning, cutting),
    sonar projectors, industrial actuators,
    piezoelectric motors
  → Note: High HV with acceptable d33 is the target for power applications

[LOW d33 + HIGH HV]:
  → Structural integrity monitoring (uses acoustic emission, not high sensitivity),
    shock sensors, military/defense transducers,
    deep-sea sonar (mechanical robustness primary concern)
```

---

### 3.3 — Tc + Hardness Combined

```
[HIGH Tc + HIGH HV]:
  → Ideal for harsh-environment deployments:
    automotive under-hood sensors, oil & gas down-hole,
    aerospace structural health monitoring,
    military sonar (shock + temperature tolerance)

[HIGH Tc + LOW HV]:
  → High-temperature sensing in mechanically gentle environments:
    gas turbine exhaust monitors (non-contact / protected),
    reactor thermal monitoring (shielded mounts)

[LOW Tc + HIGH HV]:
  → Cold-environment structural monitoring:
    cryogenic pressure sensing,
    controlled lab instruments with mechanical load
```

---

### 3.4 — All Three Properties (d33 + Tc + Hardness) — Full Matrix

```python
# Pseudo-code implementation logic

def predict_usage(d33=None, Tc=None, HV=None):
    scores = {}  # application_name: confidence_score (0-100)

    # ── MEDICAL IMAGING / BIOMEDICAL ──
    score = 0
    if d33 and d33 >= 500: score += 40
    elif d33 and d33 >= 200: score += 20
    if Tc and 100 <= Tc <= 350: score += 25  # body-safe temperature range
    if HV and HV < 600: score += 15          # soft/flexible preferred
    scores["Medical Ultrasound Imaging"] = score

    # ── IMPLANTABLE ENERGY HARVESTER ──
    score = 0
    if d33 and d33 >= 1000: score += 45
    if Tc and 100 <= Tc <= 250: score += 25
    if HV and HV < 400: score += 20          # flexible is key
    scores["Implantable Biomedical / Deep Brain Stimulation"] = score

    # ── NDT / INDUSTRIAL SENSORS ──
    score = 0
    if d33 and 50 <= d33 < 500: score += 30
    if Tc and 200 <= Tc <= 500: score += 35
    if HV and 500 <= HV <= 1000: score += 25
    scores["NDT / Industrial Condition Monitoring Sensors"] = score

    # ── HIGH-POWER ULTRASONICS (welding/cleaning) ──
    score = 0
    if d33 and d33 >= 200: score += 25
    if Tc and Tc >= 300: score += 30         # thermal stability under load
    if HV and HV >= 700: score += 35         # mechanical endurance
    scores["High-Power Ultrasonics (Welding / Cleaning)"] = score

    # ── AUTOMOTIVE SENSORS ──
    score = 0
    if d33 and 50 <= d33 < 500: score += 25
    if Tc and Tc >= 350: score += 40         # engine bay temperatures
    if HV and HV >= 600: score += 25
    scores["Automotive Sensors (Knock / Pressure / Fuel Injection)"] = score

    # ── AEROSPACE / HIGH-TEMP STRUCTURAL HEALTH MONITORING ──
    score = 0
    if d33 and d33 >= 100: score += 20
    if Tc and Tc >= 450: score += 45
    if HV and HV >= 800: score += 25
    scores["Aerospace / High-Temp Structural Health Monitoring"] = score

    # ── ENERGY HARVESTING (ambient vibration) ──
    score = 0
    if d33 and d33 >= 150: score += 40
    if Tc and Tc >= 200: score += 25
    if HV: score += 15                        # hardness less critical
    scores["Ambient Vibration Energy Harvesting"] = score

    # ── SONAR / UNDERWATER ACOUSTICS ──
    score = 0
    if d33 and 50 <= d33 <= 600: score += 30
    if Tc and Tc >= 250: score += 25
    if HV and HV >= 700: score += 35         # shock resistance underwater
    scores["Sonar / Underwater Acoustics"] = score

    # ── WEARABLE / IoT ──
    score = 0
    if d33 and d33 >= 100: score += 30
    if Tc and 100 <= Tc <= 250: score += 20
    if HV and HV < 400: score += 35          # flexible is critical
    scores["Wearable / IoT Sensors"] = score

    # ── EXTREME ENVIRONMENT (nuclear/geothermal) ──
    score = 0
    if d33 and 10 <= d33 < 100: score += 20  # low d33 is expected here
    if Tc and Tc >= 600: score += 55
    if HV and HV >= 900: score += 20
    scores["Extreme-Environment Sensing (Nuclear / Geothermal)"] = score

    # ── PRECISION ACTUATORS (MEMS / nanopositioning) ──
    score = 0
    if d33 and d33 >= 300: score += 40
    if Tc and Tc >= 200: score += 25
    if HV and 400 <= HV <= 900: score += 25
    scores["Precision Actuators / MEMS / Nanopositioning"] = score

    # Sort and return top applications
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(app, score) for app, score in ranked if score >= 30]
```

---

## PART 4 — CONFIDENCE TIER & DISPLAY LOGIC

### Recommendation Tiers

| Confidence Score | Tier      | Display Label             | Action               |
| ---------------- | --------- | ------------------------- | -------------------- |
| 70 – 100         | Primary   | "Highly Recommended" ✅   | Show prominently     |
| 45 – 69          | Secondary | "Good Fit" 🟡             | Show with brief note |
| 30 – 44          | Tertiary  | "Possible Application" ⚠️ | Show as suggestion   |
| < 30             | Excluded  | —                         | Do not show          |

### Missing Property Handling

When only 1 or 2 properties are available:

- Show confidence tier label as "Estimated" or "Partial"
- Append note: _"Prediction accuracy improves with all three properties available."_
- Weight available properties proportionally higher (scale scores for available features to equivalent of full 3-feature input)

---

## PART 5 — KNOWN REAL MATERIAL ANCHORS (for training/validation reference)

| Material         | d33 (pC/N) | Tc (°C) | HV (approx.) | Primary Applications                     |
| ---------------- | ---------- | ------- | ------------ | ---------------------------------------- |
| Quartz (SiO2)    | ~2         | 573     | ~1100        | Frequency oscillators, pressure gauges   |
| BaTiO3           | ~190       | 120     | ~500         | Early actuators, energy harvesting       |
| PZT-5A (Soft)    | ~374       | 365     | ~500         | NDT sensors, accelerometers, flow meters |
| PZT-5H (Soft)    | ~650       | ~195    | ~450         | Medical ultrasound, precision actuators  |
| PZT-4 (Hard)     | ~289       | 328     | ~700         | Sonar projectors, high-power ultrasonics |
| PZT-8 (Hard)     | ~225       | 300     | ~750         | Ultrasonic welding/cleaning motors       |
| KNN (undoped)    | ~80        | 420     | ~600         | Lead-free mid-range sensors              |
| KNN-Li (TGG)     | ~280       | 430     | ~600         | Automotive, aerospace (lead-free)        |
| PMN-PT (crystal) | ~1500–2500 | ~130    | ~400         | Medical phased array, IVUS               |
| PIN-PMN-PT       | ~1500–2700 | ~200    | ~450         | Advanced medical imaging                 |
| Sm-PMN-PT        | ~4000      | ~150    | ~400         | Implantable biomedical devices           |
| BiScO3-PbTiO3    | ~460       | 450     | ~650         | High-temp aerospace actuators            |
| BLSF ceramics    | 10–50      | >600    | ~800         | Nuclear/deep-well sensing                |
| Nd2Ti2O7 (PLS)   | <20        | >1000   | ~900         | Extreme-temp sensing (>600°C)            |

---

## PART 6 — FORMULA TYPE MODIFIER (Bulk Ceramic vs. Composite)

The user can select **Bulk Ceramic** or **Composite**. Apply these modifiers to usage scores:

### Bulk Ceramics

- Retain raw score as-is
- Note: Higher rigidity → better for sonar, NDT, high-power actuators
- Worse for: Wearables, flexible sensors

### Composites (1-3 or 0-3 connectivity)

- Boost "Wearable / IoT" and "Implantable Biomedical" scores by +15
- Boost "Underwater Acoustics / Hydrophones" by +10 (better acoustic impedance matching)
- Reduce "High-Power Ultrasonics" by −20 (mechanical fragility under load)
- Add application note: _"Composite structure improves acoustic impedance matching with biological tissue and water, making it suitable for hydrophones and medical transducers."_

---

## PART 7 — IMPLEMENTATION RECOMMENDATIONS FOR THE AGENT

### Architecture

1. **UsagePredictionEngine** — pure rule-based scoring class (no ML needed for initial version)
2. **PropertyCompleteness** tracker — knows which of {d33, Tc, HV} are present
3. **ScoreNormalizer** — adjusts scores when only partial features available
4. **ResultRenderer** — UI component showing ranked applications with confidence tiers, icons, and brief application descriptions
5. **AnchorMatcher** (optional ML) — cosine similarity to the known material table in Part 5 to find "nearest known material" and use its applications as a soft prior

### UI Suggestions

- Show top 3–5 applications only (avoid overwhelming)
- Each application card should show:
  - Application name + icon
  - Confidence tier badge
  - 1-sentence rationale: _"High d33 enables sensitive electrical signal generation suitable for…"_
  - Which properties drove the recommendation
  - Caution note if a property is marginal (e.g., Tc close to operating temperature boundary)

### Scientific Caution Notes to Display

- If Tc < 200°C: _"Low Curie temperature limits deployment to controlled temperature environments (below ~100°C)."_
- If d33 < 50 pC/N: _"Low piezoelectric coefficient restricts use to sensing; not suitable for high-displacement actuation."_
- If HV < 300: _"Low hardness indicates flexible/polymer-class material; unsuitable for rigid high-load mechanical environments."_
- If d33 is very high (>1000) but Tc is low (<150°C): _"This combination (e.g., PMN-PT class) offers exceptional sensitivity but requires thermal management for field use."_

---

## SOURCES

1. ACS Appl. Mater. Interfaces 2016, 8, 49 — KNN high d33 + high Tc
2. PMC4223717 — Piezoelectric single crystals for biomedical ultrasound transducers
3. PMC9012468 / Science Advances — Sm-PMN-PT implantable energy harvester
4. USPTO 8518291 — High temperature piezoelectric ceramics overview
5. USPTO 7658111 — Bismuth titanate high-Tc sensors
6. PMC4823784 — Nd2Ti2O7 super-stable ferroelectrics (Tc >1000°C)
7. USPTO 10756253 — Sensitivity vs. max usage temperature chart (patent summary)
8. ScienceDirect — High-electromechanical performance for high-power piezo (Vickers hardness ~9.7 GPa)
9. ScienceDirect — Dynamic fracture behavior of piezoceramics (fracture toughness 0.5–2.0 MPa·m^0.5)
10. Yujie Piezo Engineering Blog (2025) — Soft vs Hard PZT: d33, Qm, application selection
11. PI Piezo Tutorial — Curie temperature and max operating temperature rules
12. PMC9000841 — Piezoelectric materials for high-temperature applications (review)
