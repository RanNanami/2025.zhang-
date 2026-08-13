"""Generate the read-only Zhang 2025 soma/V_inh citation-chain audit.

This script deliberately writes reports and metadata only.  It never imports
or mutates the model and it never runs an experiment.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_TEXT = ROOT / "paper_text.txt"
MODEL = ROOT / "src" / "seqmem" / "model.py"
DYNAMICS = ROOT / "src" / "seqmem" / "dynamics.py"
FIG9 = ROOT / "experiments" / "fig9_strict_reproduction.py"

SOURCES = {
    "zhang_2025": "https://ira.lib.polyu.edu.hk/bitstream/10397/113864/1/Zhang_Toward_Building_Human-like.pdf",
    "zhang_2025_record": "https://ira.lib.polyu.edu.hk/handle/10397/113864",
    "zhang_delay_2020": "https://researchportal.northumbria.ac.uk/files/27753488/A_Belatreche_Elsevier_Neurocomputing.pdf",
    "sun_delay_2024": "https://backoffice.biblio.ugent.be/download/01JHQKDXSJFR0PG1DZR7MXC4GB/01JHQKKZEHGCADP620DMTB42AD",
    "htm_2011": "https://numenta.com/assets/pdf/whitepapers/hierarchical-temporal-memory-cortical-learning-algorithm-0.2.1-en.pdf",
    "cui_htm_2016": "https://www.cortical.io/static/downloads/continuous-online-sequence-learning.pdf",
    "zhang_minicolumn_2024": "https://doi.org/10.1109/TNNLS.2022.3213688",
    "lan_minicolumn_2021": "https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.650430/full",
}


def line_number(path: Path, pattern: str) -> int | None:
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if pattern in line:
            return index
    return None


def code_value(path: Path, pattern: str, default: str = "NOT_FOUND") -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(pattern, text)
    return match.group(1) if match else default


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def main() -> None:
    if not PAPER_TEXT.exists():
        raise FileNotFoundError(PAPER_TEXT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "results" / f"citation_chain_soma_vinh_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    # Zhang 2025 source map.  Printed page numbers refer to the IEEE version.
    source_rows = [
        {"item": "V_pro", "paper_page": "10146", "section_equation": "II-B.3", "paper_wording": "proximal feedforward input directly contributes to soma potential", "numeric_value_given": "No", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "V_api", "paper_page": "10146", "section_equation": "II-B.3", "paper_wording": "apical feedback input directly contributes to soma potential", "numeric_value_given": "No", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "V_dis", "paper_page": "10146/10149", "section_equation": "II-B.2, II-E", "paper_wording": "dendritic spike raises distal input to v_dep and creates predictive state", "numeric_value_given": "v_dep=0.5", "reference_adjacent": "[35]-[37]", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "V_inh", "paper_page": "10147", "section_equation": "II-B.5, Eq.(3)", "paper_wording": "inhibitory feedback from other mini-columns", "numeric_value_given": "No", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "STILL_UNKNOWN"},
        {"item": "V_osc", "paper_page": "10147", "section_equation": "II-B.4, Eq.(2)", "paper_wording": "layer-dependent subthreshold oscillation defined as a sine wave", "numeric_value_given": "A_osc=0.5; f bound; phi_0=0 or pi", "reference_adjacent": "[38]-[42]", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "eta", "paper_page": "10147", "section_equation": "II-B.5", "paper_wording": "refractory kernel is -theta after firing in the current cycle and zero otherwise", "numeric_value_given": "-theta", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "theta", "paper_page": "10150", "section_equation": "Table I", "paper_wording": "firing threshold of neurons", "numeric_value_given": "1.0", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "theta_d", "paper_page": "10150", "section_equation": "Table I", "paper_wording": "firing threshold of distal dendrites", "numeric_value_given": "1.0", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "phase_precession", "paper_page": "10147/10149", "section_equation": "II-B.4, II-E", "paper_wording": "depolarization shifts oscillation to the trough; oscillation returns to phi_0 after prediction", "numeric_value_given": "half-cycle depolarized duration", "reference_adjacent": "[26]-[27], [39]-[42]", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "oscillation_frequency", "paper_page": "10147/10150", "section_equation": "Eq.(2), Table I", "paper_wording": "frequency decreases in higher layers and is bounded relative to K", "numeric_value_given": "f_osc < 1/(2K)", "reference_adjacent": "[23]-[25], [38]", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "oscillation_amplitude", "paper_page": "10150", "section_equation": "Table I", "paper_wording": "amplitude of neural oscillation", "numeric_value_given": "0.5", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "synaptic_response_kernel", "paper_page": "10146", "section_equation": "Eq.(1)", "paper_wording": "double exponential V0[exp(-t/tau_m)-exp(-t/tau_s)]", "numeric_value_given": "V0/tau values absent", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "dendritic_spike", "paper_page": "10145/10146", "section_equation": "II-A, II-B.1", "paper_wording": "distal potential reaching theta_d triggers local dendritic spike and reset", "numeric_value_given": "theta_d=1.0", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "soma_spike", "paper_page": "10147", "section_equation": "II-B.5, Eq.(3)", "paper_wording": "largest membrane potential within mini-column and above theta fires", "numeric_value_given": "theta=1.0", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "intracolumn_inhibition", "paper_page": "10147", "section_equation": "II-B.5", "paper_wording": "constant large enough to suppress other neurons until cycle end", "numeric_value_given": "Exact amplitude not required", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "intercolumn_inhibition", "paper_page": "10147", "section_equation": "Eq.(3)", "paper_wording": "feedback from other mini-columns", "numeric_value_given": "No", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "STILL_UNKNOWN"},
        {"item": "refractory", "paper_page": "10147", "section_equation": "II-B.5", "paper_wording": "fired neuron cannot fire again in current cycle", "numeric_value_given": "eta=-theta after firing", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
        {"item": "winner_firing", "paper_page": "10145/10147", "section_equation": "II-A, II-B.5", "paper_wording": "predictive neuron fires first and suppresses same-column peers; no predictive neuron causes burst", "numeric_value_given": "No", "reference_adjacent": "None", "explicit_inheritance": "No", "status": "PAPER_EXPLICIT"},
    ]
    write_csv(output / "ZHANG_SOMA_SOURCE_MAP.csv", list(source_rows[0]), source_rows)
    write_text(output / "ZHANG_SOMA_SOURCE_MAP.md", """# Zhang 2025 Soma Source Map\n\nThis table is based on the supplied local Zhang et al. 2025 PDF and its printed page numbers. The paper gives the semantic placement of `V_inh` in Eq.(3), but not a computable continuous intercolumn inhibitory function. Table I is read visually because text extraction loses the table values.\n\n| Item | Paper page | Section/equation | Numeric value | Adjacent references | Status |\n|---|---:|---|---|---|---|\n""" + "\n".join(f"| {r['item']} | {r['paper_page']} | {r['section_equation']} | {r['numeric_value_given']} | {r['reference_adjacent']} | {r['status']} |" for r in source_rows) + "\n\nThe local PDF is `C:/Users/苒苒/xwechat_files/wxid_h6mi9wbc2oh822_70f4/msg/file/2026-05/Zhang 等 - 2025 - Toward Building Human-Like Sequential Memory Using Brain-Inspired Spiking Neural Models.pdf`.\n""")

    citation_rows = [
        ["20", "synaptic delay plasticity is listed among forms of plasticity", "method background", "No", "delay learning only", "high"],
        ["21", "synaptic delay plasticity is listed among forms of plasticity", "method background", "No", "delay learning only", "high"],
        ["22", "structural plasticity is listed among forms of plasticity", "biological motivation", "No", "structural plasticity only", "medium"],
        ["23-25", "neural oscillation is described as a biological mechanism", "biology", "No", "oscillation motivation", "high"],
        ["26-27", "phase precession is described as a neurophysiological process", "biology", "No", "phase-precession motivation", "high"],
        ["28-29", "synchronized neural activity is listed among mechanisms", "biology", "No", "synchronization motivation", "medium"],
        ["33", "architecture is inspired by neocortex structure and computational properties", "HTM/neocortex semantics", "No", "hierarchy, mini-columns and predictive semantics; not Eq.(3) V_inh", "high"],
        ["34", "predictive/depolarized state is followed by a citation to subthreshold oscillations", "cellular biology", "No", "oscillation/depolarization motivation", "high"],
        ["35-37", "depolarized state lasts half a cycle and is supported by dendritic literature", "dendritic biology", "No", "dendritic spike motivation", "high"],
        ["38", "oscillation is introduced as a pacemaker mechanism", "biology", "No", "pacemaker motivation", "medium"],
        ["39-42", "phase precession and oscillatory timing are discussed", "biology", "No", "phase-precession motivation", "high"],
        ["58", "baseline methods and Fig.7/Fig.9 comparison source", "benchmark/method", "No", "HTM benchmark provenance; not soma dynamics", "high"],
        ["62", "CBT data set source", "dataset", "No", "dataset only", "high"],
        ["64-65", "LUMAKG/ANAKG comparison methods", "benchmark", "No", "comparison methods only", "high"],
    ]
    write_csv(output / "CITATION_CONTEXT_MAP.csv", ["reference_number", "Zhang_sentence_context", "method_or_biology", "explicit_follow_use_adopt", "likely_parameter_source", "priority"], [dict(zip(["reference_number", "Zhang_sentence_context", "method_or_biology", "explicit_follow_use_adopt", "likely_parameter_source", "priority"], row)) for row in citation_rows])

    write_text(output / "SOMA_EQUATION_LINEAGE.md", f"""# Soma Equation Lineage\n\n## Zhang 2025\n\nZhang Eq.(3), printed p.10147, defines the soma potential as the sum of `V_pro`, `V_api`, `V_dis`, `V_inh`, `V_osc`, and the refractory term `eta`. The paper states the winner condition and same-column suppression, but does not define a formula for `V_inh^l(t)`.\n\n## Candidate technical sources\n\n| Candidate | What it actually provides | Lineage judgment |\n|---|---|---|\n| Zhang et al. 2020, synaptic delay-weight plasticity | supervised delay/weight learning for SNNs; no evidence in the accessible metadata that it defines Zhang 2025's three-zone DS-neuron or intercolumn feedback | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |\n| Sun et al. 2024, delay learning based on temporal coding | a different standard SNN/Slayer-style model with a PSP and refractory kernel; it has no `V_inh` term and is not cited by Zhang 2025 as the Eq.(3) source | `CONCEPTUAL_SIMILARITY` |\n| HTM 2011 white paper / Cui et al. 2016 | predictive cells, bursting, sparse temporal memory and sequence prediction semantics; no continuous soma voltage equation corresponding to Zhang Eq.(3) | `HTM_SEMANTIC_ONLY_NOT_CONTINUOUS_VINH_SOURCE` |\n| Lan et al. 2021 minicolumn SNN | global inhibitory interneuron and stronger-lateral-input winner semantics, but a different model and not a cited inheritance source in Zhang 2025 | `AUTHOR_PREVIOUS_WORK_CANDIDATE` |\n\nNo source met the required standard of “Zhang explicitly follows/adopts this source and that source defines the same `V_inh` function.” Therefore the lineage is `NO_MATCH` for an exact recoverable soma equation.\n\nSources: [Zhang 2025]({SOURCES['zhang_2025']}), [Zhang et al. 2020]({SOURCES['zhang_delay_2020']}), [Sun et al. 2024]({SOURCES['sun_delay_2024']}), [HTM 2011]({SOURCES['htm_2011']}), [Lan et al. 2021]({SOURCES['lan_minicolumn_2021']}).\n""")

    write_text(output / "PSP_NORMALIZATION_LINEAGE.md", f"""# PSP Normalization Lineage\n\n| Layer | Formula/value | Evidence class |\n|---|---|---|\n| Zhang 2025 Eq.(1) | `kappa(t) = V0 [exp(-t/tau_m) - exp(-t/tau_s)]` | `PAPER_EXPLICIT` formula, numeric scale unresolved |\n| Zhang 2025 Table I | `V0`, `tau_m`, and `tau_s` are not given as numeric rows | `STILL_UNKNOWN` |\n| Zhang 2025 surrounding text | `tau_m` and `tau_s` jointly govern kernel shape | `PAPER_EXPLICIT` qualitative statement |\n| Current `src/seqmem/dynamics.py` | when `response_scale is None`, `kernel_scale = 1 / unscaled_peak`, so the peak is normalized to one | `LOCAL_IMPLEMENTATION` |\n| Current Fig.9 strict | `response_scale=None` by default | `LOCAL_IMPLEMENTATION` / not paper-proven |\n| Zhang et al. 2020 | cited by Zhang 2025 for delay-weight plasticity, not explicitly for Eq.(1) scaling | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |\n| Sun et al. 2024 | uses a different PSP expression and explicitly describes its own time constants; no Zhang inheritance statement | `CONCEPTUAL_SIMILARITY` |\n\nConclusion: PSP scaling is **partially recoverable** at the formula level but not at the numeric `V0`/time-constant level. The current normalized kernel is a local implementation assumption, not a defensible Zhang parameter. It must not be promoted to paper truth without author confirmation or an explicit source sentence.\n\nSources: [Zhang 2025]({SOURCES['zhang_2025']}), [Zhang et al. 2020]({SOURCES['zhang_delay_2020']}), [Sun et al. 2024]({SOURCES['sun_delay_2024']}).\n""")

    evidence_rows = [
        ["V_inh kernel", "semantic only", "Zhang 2025 Eq.(3), p.10147", "inhibitory feedback from other mini-columns", "No inheritance sentence", "low", "STILL_UNKNOWN"],
        ["V_inh amplitude", "not given", "Zhang 2025 Eq.(3)", "not specified", "No", "high", "STILL_UNKNOWN"],
        ["V_inh duration", "not given", "Zhang 2025 Eq.(3)", "not specified", "No", "high", "STILL_UNKNOWN"],
        ["V_inh spatial scope", "other mini-columns", "Zhang 2025 Eq.(3)", "column-level wording, neuron-level formula unresolved", "No", "medium", "PAPER_EXPLICIT"],
        ["intracolumn suppression", "constant sufficiently large", "Zhang 2025 II-B.5", "suppressed peers cannot fire until cycle end", "No", "high", "PAPER_EXPLICIT_REFERENCE_DEFINED"],
        ["refractory amplitude", "-theta after firing", "Zhang 2025 II-B.5", "eta(t-t_fire)=-theta in current cycle", "No", "high", "PAPER_EXPLICIT"],
        ["refractory duration", "current cycle", "Zhang 2025 II-B.5", "otherwise eta=0", "No", "high", "PAPER_EXPLICIT"],
        ["tau_m", "not numeric", "Zhang 2025 Eq.(1)", "shape parameter only", "No", "high", "STILL_UNKNOWN"],
        ["tau_s", "not numeric", "Zhang 2025 Eq.(1)", "shape parameter only", "No", "high", "STILL_UNKNOWN"],
        ["V0", "not numeric", "Zhang 2025 Eq.(1)", "scale parameter only", "No", "high", "STILL_UNKNOWN"],
        ["PSP normalization", "not stated", "Zhang 2025 Eq.(1)", "no peak-normalization sentence", "No", "high", "STILL_UNKNOWN"],
        ["v_dep", "0.5", "Zhang 2025 Table I", "depolarization potential", "No", "high", "PAPER_EXPLICIT"],
        ["A_osc", "0.5", "Zhang 2025 Table I", "oscillation amplitude", "No", "high", "PAPER_EXPLICIT"],
        ["f_osc", "<1/(2K), layer-dependent", "Zhang 2025 Table I and II-B.4", "frequency decreases in higher layers", "No", "high", "PAPER_EXPLICIT"],
        ["phi_0", "0 or pi", "Zhang 2025 Table I and III-A", "initial phase; adjacent layer difference pi", "No", "high", "PAPER_EXPLICIT"],
        ["theta", "1.0", "Zhang 2025 Table I", "firing threshold", "No", "high", "PAPER_EXPLICIT"],
        ["theta_d", "1.0", "Zhang 2025 Table I", "distal dendrite threshold", "No", "high", "PAPER_EXPLICIT"],
        ["w0", "0.5", "Zhang 2025 Table I", "initial synaptic weight", "No", "high", "PAPER_EXPLICIT"],
        ["delta_w", "0.01", "Zhang 2025 Table I", "weight adjustment", "No", "high", "PAPER_EXPLICIT"],
        ["delta_w_prime", "0.01", "Zhang 2025 Table I", "incorrect-prediction weight reduction", "No", "high", "PAPER_EXPLICIT"],
        ["dt", "not given", "Zhang 2025", "continuous formula, no integration grid specified", "No", "high", "STILL_UNKNOWN"],
        ["simultaneous spike handling", "not given", "Zhang 2025", "no tie policy specified", "No", "high", "STILL_UNKNOWN"],
    ]
    write_csv(output / "REFERENCE_EVIDENCE_TABLE.csv", ["parameter_or_rule", "Zhang_status", "reference", "reference_page_or_equation", "value_or_formula", "inheritance_evidence", "confidence", "classification"], [dict(zip(["parameter_or_rule", "Zhang_status", "reference", "reference_page_or_equation", "value_or_formula", "inheritance_evidence", "confidence", "classification"], row)) for row in evidence_rows])

    model_line = line_number(MODEL, "def spike_response")
    norm_line = line_number(DYNAMICS, "return 1.0 if response_scale is None")
    tau_line = line_number(DYNAMICS, "tau_m: float = 0.10")
    fig9_line = line_number(FIG9, "response_scale: float | None = None")
    parity_rows = [
        ["continuous V_inh", "no public formula", "no continuous intercolumn V_inh implementation", "src/seqmem/model.py: prediction selector and diagnostics", "UNKNOWN", "Eq.(3) semantic slot only"],
        ["intracolumn suppression", "constant sufficiently large", "boolean intracolumn selector; one selected candidate per column", f"src/seqmem/model.py:{line_number(MODEL, 'intracolumn_inhibition')}", "PARTIAL", "semantic parity, numeric mechanism not represented"],
        ["soma winner", "maximum full membrane potential in column", "candidate/event score selection; no full soma component trace", f"src/seqmem/model.py:{line_number(MODEL, 'best_by_event')}", "UNPROVEN", "candidate score is not proven soma potential"],
        ["refractory eta", "-theta after firing in current cycle", "DSNeuronState returns -inf while refractory", f"src/seqmem/dynamics.py:{line_number(DYNAMICS, 'if self.is_refractory')}", "PARTIAL", "same no-refire semantics, different numeric term"],
        ["tau_m", "not numerically specified", "0.10", f"src/seqmem/dynamics.py:{tau_line}", "UNPROVEN", "local default"],
        ["tau_s", "not numerically specified", "0.02", f"src/seqmem/dynamics.py:{line_number(DYNAMICS, 'tau_s: float = 0.02')}", "UNPROVEN", "local default"],
        ["V0/PSP scale", "formula has V0; numeric/normalization unspecified", "response_scale=None normalizes unscaled peak to 1", f"src/seqmem/dynamics.py:{norm_line}", "UNPROVEN", "local normalization assumption"],
        ["v_dep", "0.5", "0.5", f"src/seqmem/dynamics.py:{line_number(DYNAMICS, 'v_dep: float = 0.5')}", "MATCH", "paper Table I"],
        ["A_osc", "0.5", "0.5", f"src/seqmem/dynamics.py:{line_number(DYNAMICS, 'oscillation_amplitude: float = 0.5')}", "MATCH", "paper Table I"],
        ["theta/theta_d", "1.0/1.0", "1.0/1.0", f"src/seqmem/dynamics.py:{line_number(DYNAMICS, 'dendrite_threshold: float = 1.0')}", "MATCH", "paper Table I"],
        ["dt", "not specified", "0.005", f"src/seqmem/model.py:{line_number(MODEL, 'integration_step: float = 0.005')}", "UNPROVEN", "local numerical integration choice"],
        ["phase precession", "shift to trough and reset", "-cos after depolarization; reset via state clearing", f"src/seqmem/dynamics.py:{line_number(DYNAMICS, 'return -params.oscillation_amplitude * math.cos')}", "PARTIAL", "qualitative semantic match, exact numeric rule not inherited"],
        ["firing time", "actual soma threshold crossing", "continuous candidate solver after dendritic crossing", f"src/seqmem/model.py:{line_number(MODEL, 'state.membrane_potential')}", "PARTIAL", "no V_inh and no population WTA"],
    ]
    write_csv(output / "REFERENCE_TO_CODE_PARITY.csv", ["item", "paper_or_reference_supported_value", "current_code_value", "current_source", "status", "notes"], [dict(zip(["item", "paper_or_reference_supported_value", "current_code_value", "current_source", "status", "notes"], row)) for row in parity_rows])

    write_text(output / "VINH_CITATION_CHAIN_REPORT.md", f"""# V_inh Citation-Chain Report\n\n## Question-by-question answers\n\n| # | Question | Answer | Evidence class |\n|---:|---|---|---|\n| 1 | What is the source of each Eq.(3) term? | `V_pro`, `V_api`, `V_dis`, `V_inh`, `V_osc`, and `eta` are defined in Zhang 2025 Eq.(3); only `V_inh` lacks a computable definition. | `PAPER_EXPLICIT` / `STILL_UNKNOWN` |\n| 2 | Which terms are directly defined? | The three input zones, oscillation term, refractory term, winner condition, burst condition, and same-column suppression semantics. | `PAPER_EXPLICIT` |\n| 3 | Which terms come from references? | References provide motivation or adjacent mechanisms; no citation is explicitly declared as the complete Eq.(3) implementation source. | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |\n| 4 | Which are biological motivation only? | Neocortex/HTM structure, oscillations, phase precession, dendritic spike biology, synchronization and pyramidal-cell inspiration. | `BIOLOGICAL_MOTIVATION_ONLY` |\n| 5 | Is there a concrete `V_inh(t)` formula? | No. | `STILL_UNKNOWN` |\n| 6 | Which paper provides it? | None found in the audited Zhang citation chain. | `STILL_UNKNOWN` |\n| 7 | Does Zhang explicitly inherit it? | No explicit following/adopting/using sentence was found. | `STILL_UNKNOWN` |\n| 8 | Is `V_inh` amplitude given? | No. | `STILL_UNKNOWN` |\n| 9 | Is its temporal kernel given? | No. | `STILL_UNKNOWN` |\n| 10 | Is its spatial scope given? | Only `from other mini-columns` is stated; neighborhood and neuron/column aggregation are not defined. | `PAPER_EXPLICIT` / `STILL_UNKNOWN` |\n| 11 | What is the source of `tau_m`? | Zhang gives its qualitative role in Eq.(1), not a numeric value or inherited source. | `STILL_UNKNOWN` |\n| 12 | What is the source of `tau_s`? | Same as `tau_m`. | `STILL_UNKNOWN` |\n| 13 | What is the source of `V0`? | It appears in the paper formula, but no numeric value or normalization rule is supplied. | `STILL_UNKNOWN` |\n| 14 | Is PSP normalized? | The paper does not say so; current peak normalization is local code behavior. | `STILL_UNKNOWN` / `IMPLEMENTATION_ASSUMPTION` |\n| 15 | Why is current code normalized? | `response_scale=None` uses `1 / unscaled_peak` in `src/seqmem/dynamics.py`; this is a local implementation choice from the reproduction work. | `IMPLEMENTATION_ASSUMPTION` |\n| 16 | Does author previous work match? | The accessible delay-learning work concerns delay/weight learning, not a proven inheritance of Zhang's three-zone DS-neuron and `V_inh`. | `REFERENCE_EXPLICIT_BUT_INHERITANCE_UNPROVEN` |\n| 17 | What is the source of oscillation amplitude? | Table I explicitly gives `A_osc=0.5`; adjacent papers motivate oscillations but do not supply an inherited value. | `PAPER_EXPLICIT` |\n| 18 | What is the source of frequency? | Table I gives a layer-dependent bound `f_osc < 1/(2K)` and the paper says frequency decreases up the hierarchy. | `PAPER_EXPLICIT` |\n| 19 | What is the numeric phase-precession rule source? | The paper describes a shift to the trough, half-cycle predictive duration and reset, but not a complete numeric state equation. | `PAPER_EXPLICIT` / `STILL_UNKNOWN` |\n| 20 | Is refractory fully paper-explicit? | Its current-cycle no-refire semantics and `eta=-theta` behavior are explicit; exact implementation discretization is not. | `PAPER_EXPLICIT` |\n| 21 | Does intracolumn inhibition need a concrete strength? | No. Zhang says it is a constant large enough to prevent suppressed firing. | `PAPER_EXPLICIT_REFERENCE_DEFINED` |\n| 22 | Can soma winner be fully implemented? | Not from the public chain alone: the winner requires the largest full soma potential, including unresolved `V_inh`. | `UNPROVEN` |\n| 23 | Is current event selection paper-proven? | No. Candidate score/event selection is not proven equivalent to full soma WTA. | `UNPROVEN` |\n| 24 | Was a clear current-code mismatch found? | Yes as an evidence gap: normalized PSP, local candidate scoring and absent continuous `V_inh` are not paper-proven. A unique corrective formula was not recovered. | `UNKNOWN` |\n| 25 | What can be safely repaired? | Documentation, provenance labels, and audit traces. No model equation can be safely repaired from this chain alone. | `IMPLEMENTATION_ASSUMPTION` |\n| 26 | What can only be candidate ablation? | Alternative PSP scaling, candidate ranking and inhibition shapes. | `AUTHOR_PREVIOUS_WORK_CANDIDATE` |\n| 27 | What must wait for the author? | The numeric `V_inh(t)` function, `V0`, `tau_m`, `tau_s`, normalization and exact soma-WTA implementation. | `STILL_UNKNOWN` |\n| 28 | Should the model be changed now? | No. | `DO_NOT_MODIFY_MODEL_WITHOUT_MORE_EVIDENCE` |\n| 29 | Primary conclusion? | `CONTINUOUS_VINH_PUBLICLY_UNDERSPECIFIED` | final |\n| 30 | Recommended next step? | `DO_NOT_MODIFY_MODEL_WITHOUT_MORE_EVIDENCE` | final |\n\n## Citation-chain buckets\n\n### Bucket A - recovered\n\n- Eq.(1) double-exponential PSP form.\n- `v_dep=0.5`, `A_osc=0.5`, `theta=1.0`, `theta_d=1.0`, `w0=0.5`, `delta_w=0.01`, `delta_w_prime=0.01`.\n- qualitative half-cycle depolarization, phase shift to trough, current-cycle refractory semantics, and constant-sufficiently-large intracolumn suppression.\n\n### Bucket B - candidate only\n\n- HTM predictive/burst/winner semantics from [33] and [58].\n- global inhibitory interneuron semantics in the author-associated 2021 minicolumn SNN work.\n- delay-learning details in [20]/[21].\n- PSP/refractory equations in other SNN papers.\n\n### Bucket C - unrecoverable\n\n- continuous intercolumn `V_inh(t)` formula and all of its numerical parameters.\n- numeric `V0`, `tau_m`, `tau_s`, PSP normalization choice.\n- exact population-level soma WTA implementation and simultaneous-spike policy.\n\n## HTM decision\n\nThe HTM white paper defines predictive cells, bursting and temporal sequence algorithms, but it is not a source for a continuous soma voltage or `V_inh(t)` equation. Marked as `HTM_SEMANTIC_ONLY_NOT_CONTINUOUS_VINH_SOURCE`.\n\n## No model repair in this round\n\nNo implementation was changed. No parameter sweep or Fig.8/Fig.9 formal run was performed.\n\nSources: [Zhang 2025]({SOURCES['zhang_2025']}), [HTM 2011]({SOURCES['htm_2011']}), [Cui et al. 2016]({SOURCES['cui_htm_2016']}), [Zhang et al. 2020]({SOURCES['zhang_delay_2020']}), [Sun et al. 2024]({SOURCES['sun_delay_2024']}), [Lan et al. 2021]({SOURCES['lan_minicolumn_2021']}).\n""")

    summary = {
        "audit_timestamp": timestamp,
        "git_commit_sha_at_generation": git_sha(),
        "paper_source": SOURCES["zhang_2025"],
        "continuous_vinh": {"status": "PUBLICLY_UNDERSPECIFIED", "source": "Zhang 2025 Eq.(3) semantic slot; no public formula recovered", "recoverable": False},
        "intracolumn_inhibition": {"status": "PAPER_EXPLICIT_REFERENCE_DEFINED", "source": "Zhang 2025 II-B.5", "value": "constant large enough; exact numeric amplitude not required"},
        "tau_m": {"status": "STILL_UNKNOWN", "source": "Zhang 2025 Eq.(1) gives shape role only", "current_code": 0.10},
        "tau_s": {"status": "STILL_UNKNOWN", "source": "Zhang 2025 Eq.(1) gives shape role only", "current_code": 0.02},
        "V0": {"status": "STILL_UNKNOWN", "source": "Zhang 2025 Eq.(1) formula only", "current_code": "normalized peak when response_scale=None"},
        "psp_normalization": {"status": "PARTIAL", "source": "formula explicit, normalization not stated", "current_code": "peak normalized to 1 by local default"},
        "oscillation_amplitude": {"status": "PAPER_EXPLICIT", "value": 0.5},
        "oscillation_frequency": {"status": "PAPER_EXPLICIT", "value": "layer-dependent; f_osc < 1/(2K)"},
        "phase_precession": {"status": "PARTIAL", "value": "shift to trough, predictive state lasts half cycle, reset to phi_0"},
        "refractory": {"status": "PAPER_EXPLICIT", "value": "eta=-theta after firing in current cycle"},
        "soma_winner": {"status": "UNPROVEN", "paper_rule": "largest full soma membrane potential in column", "current_rule": "candidate/event selection"},
        "current_code_major_mismatch": "No uniquely recoverable numeric V_inh formula; current normalized PSP and candidate-score WTA are unproven assumptions rather than paper-derived values.",
        "primary_conclusion": "CONTINUOUS_VINH_PUBLICLY_UNDERSPECIFIED",
        "recommended_next_step": "DO_NOT_MODIFY_MODEL_WITHOUT_MORE_EVIDENCE",
        "formal_experiments_run": False,
        "tests_run": {"compileall": "pending", "full_unittest": "pending", "git_diff_check": "pending"},
        "sources": SOURCES,
    }
    (output / "FINAL_CITATION_CHAIN_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
