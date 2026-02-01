from __future__ import annotations

"""Built-in FTIR correlation library (v2).

Pure data module: no UI imports, no file I/O.
Ranges are in cm^-1.

Notes
- This is a pragmatic, broad library for automated suggestions.
- Many bands overlap; the matcher is intentionally conservative.
"""

from typing import Any, Dict, List, Tuple

FTIR_LIBRARY_VERSION = "2.0"

# Allowed vocab per spec
_ALLOWED_SHAPES = {"sharp", "medium", "broad"}
_ALLOWED_INTENSITIES = {"weak", "medium", "strong", "variable"}


def _entry(
    *,
    id: str,
    range_cm1: Tuple[float, float],
    label: str,
    typical_shape: List[str],
    typical_intensity: List[str],
    notes: str,
    positive: List[Dict[str, Any]] | None = None,
    negative: List[Dict[str, Any]] | None = None,
    examples: List[float] | None = None,
) -> Dict[str, Any]:
    ts = [str(s).strip().lower() for s in (typical_shape or [])]
    ti = [str(s).strip().lower() for s in (typical_intensity or [])]
    if any(s not in _ALLOWED_SHAPES for s in ts):
        raise ValueError(f"Invalid typical_shape in {id}: {typical_shape}")
    if any(s not in _ALLOWED_INTENSITIES for s in ti):
        raise ValueError(f"Invalid typical_intensity in {id}: {typical_intensity}")

    lo, hi = float(range_cm1[0]), float(range_cm1[1])
    if lo > hi:
        lo, hi = hi, lo

    return {
        "id": str(id),
        "range_cm1": (lo, hi),
        "label": str(label),
        "typical_shape": ts,
        "typical_intensity": ti,
        "notes": str(notes),
        "context_hints": {
            "positive": list(positive or []),
            "negative": list(negative or []),
        },
        "examples": list(examples or []),
    }


# Correlation library v2
# Ranges/notes are broadly consistent with common IR tables (Chemistry LibreTexts;
# CSU Stanislaus IR characteristic frequencies). This is not exhaustive.
FTIR_LIBRARY_V2: List[Dict[str, Any]] = [
    # --- O-H / N-H region ---
    _entry(
        id="oh_free_alcohol_phenol",
        range_cm1=(3700, 3580),
        label="O–H stretch (free alcohol/phenol)",
        typical_shape=["sharp"],
        typical_intensity=["weak", "medium", "variable"],
        notes="Free (non H-bonded) O–H; shifts lower and broadens with H-bonding.",
        positive=[{"range_cm1": (1260, 1000), "text": "C–O stretch often present (alcohol/phenol)"}],
        negative=[{"range_cm1": (3300, 2500), "text": "Very broad COOH O–H suggests acid instead"}],
        examples=[3640.0],
    ),
    _entry(
        id="oh_hbond_alcohol_phenol",
        range_cm1=(3600, 3200),
        label="O–H stretch (H-bonded alcohol/phenol)",
        typical_shape=["broad"],
        typical_intensity=["strong", "variable"],
        notes="Broad due to H-bonding; stronger and broader with more H-bonding/water.",
        positive=[{"range_cm1": (1260, 1000), "text": "C–O stretch often present"}],
        negative=[{"range_cm1": (2260, 2220), "text": "C≡N is unrelated; overlap unlikely"}],
        examples=[3400.0],
    ),
    _entry(
        id="oh_carboxylic_acid_very_broad",
        range_cm1=(3300, 2500),
        label="O–H stretch (carboxylic acid, very broad)",
        typical_shape=["broad"],
        typical_intensity=["strong"],
        notes="Very broad ‘tongue’ from ~3300–2500; often with strong C=O near ~1710–1760 (variable).",
        positive=[{"range_cm1": (1760, 1680), "text": "C=O stretch often present for acids"}],
        negative=[{"range_cm1": (3700, 3580), "text": "Free sharp O–H suggests alcohol/phenol instead"}],
        examples=[3000.0],
    ),
    _entry(
        id="nh_primary_secondary_stretch",
        range_cm1=(3500, 3300),
        label="N–H stretch (amine/amide)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["weak", "medium", "variable"],
        notes="Primary amines often show two bands; amides can be broader and overlap O–H region.",
        positive=[{"range_cm1": (1650, 1580), "text": "Amide II / N–H bend region support"}],
        negative=[{"range_cm1": (3300, 2500), "text": "Very broad COOH O–H suggests acid instead"}],
        examples=[3350.0, 3450.0],
    ),
    _entry(
        id="amide_ii_nh_bend",
        range_cm1=(1650, 1580),
        label="Amide II / N–H bend + C–N stretch",
        typical_shape=["medium"],
        typical_intensity=["medium", "strong", "variable"],
        notes="Often accompanies amide I (C=O) ~1690–1630; overlaps aromatic C=C.",
        positive=[{"range_cm1": (1690, 1630), "text": "Amide I (C=O) support"}, {"range_cm1": (3500, 3300), "text": "N–H stretch support"}],
        negative=[],
        examples=[1550.0, 1620.0],
    ),

    # --- C-H stretches ---
    _entry(
        id="ch_sp2_stretch",
        range_cm1=(3100, 3000),
        label="=C–H stretch (sp2)",
        typical_shape=["sharp"],
        typical_intensity=["weak", "medium"],
        notes="Aromatic/alkene C–H stretches just above 3000.",
        positive=[{"range_cm1": (1620, 1450), "text": "Aromatic/alkene C=C stretch often present"}],
        negative=[],
        examples=[3030.0],
    ),
    _entry(
        id="ch_sp3_stretch",
        range_cm1=(3000, 2840),
        label="C–H stretch (sp3)",
        typical_shape=["sharp"],
        typical_intensity=["medium", "strong", "variable"],
        notes="Alkyl C–H stretches ~2960/2920/2870/2850.",
        positive=[{"range_cm1": (1470, 1350), "text": "CH bending region supports alkyl"}],
        negative=[],
        examples=[2960.0, 2920.0, 2850.0],
    ),
    _entry(
        id="ch_aldehyde_doublet",
        range_cm1=(2830, 2695),
        label="Aldehyde C–H stretch (Fermi doublet)",
        typical_shape=["sharp"],
        typical_intensity=["weak", "medium"],
        notes="Often a pair near ~2820 and ~2720; supports aldehyde C=O assignment.",
        positive=[{"range_cm1": (1740, 1720), "text": "Aldehyde C=O often present"}],
        negative=[],
        examples=[2820.0, 2720.0],
    ),

    # --- Triple bonds ---
    _entry(
        id="nitrile_cn",
        range_cm1=(2260, 2220),
        label="C≡N stretch (nitrile)",
        typical_shape=["sharp"],
        typical_intensity=["medium", "strong"],
        notes="Usually sharp; conjugation can shift slightly lower.",
        positive=[],
        negative=[],
        examples=[2240.0],
    ),
    _entry(
        id="alkyne_cc",
        range_cm1=(2260, 2100),
        label="C≡C stretch (alkyne)",
        typical_shape=["sharp"],
        typical_intensity=["weak", "variable"],
        notes="Often weak; terminal alkynes may show C–H near ~3300.",
        positive=[{"range_cm1": (3330, 3260), "text": "Terminal alkyne C–H may appear near ~3300"}],
        negative=[],
        examples=[2150.0],
    ),

    # --- Aromatic overtones ---
    _entry(
        id="aromatic_overtones",
        range_cm1=(2000, 1600),
        label="Aromatic overtones/combination bands",
        typical_shape=["broad", "medium"],
        typical_intensity=["weak", "variable"],
        notes="Weak broad/structured bands; supports aromatic ring when seen with 1600/1500 cm^-1 bands.",
        positive=[{"range_cm1": (1620, 1585), "text": "Aromatic C=C (~1600) support"}, {"range_cm1": (1515, 1450), "text": "Aromatic C=C (~1500) support"}],
        negative=[],
        examples=[],
    ),

    # --- Carbonyl region (C=O) ---
    _entry(
        id="carbonyl_ester",
        range_cm1=(1760, 1730),
        label="C=O stretch (ester)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong"],
        notes="Ester C=O typically ~1750–1735; conjugation lowers (can approach ~1715).",
        positive=[{"range_cm1": (1300, 1000), "text": "C–O stretch (ester/ether) often strong"}],
        negative=[{"range_cm1": (3300, 2500), "text": "Very broad O–H suggests carboxylic acid"}],
        examples=[1740.0],
    ),
    _entry(
        id="carbonyl_carboxylic_acid",
        range_cm1=(1725, 1680),
        label="C=O stretch (carboxylic acid)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong"],
        notes="Often ~1710; conjugation lowers; pairs with very broad acid O–H (3300–2500).",
        positive=[{"range_cm1": (3300, 2500), "text": "Acid O–H very broad supports"}],
        negative=[],
        examples=[1710.0],
    ),
    _entry(
        id="carbonyl_aldehyde",
        range_cm1=(1740, 1720),
        label="C=O stretch (aldehyde)",
        typical_shape=["sharp"],
        typical_intensity=["strong"],
        notes="Often overlaps ketone/ester; aldehyde C–H doublet (2830–2695) supports.",
        positive=[{"range_cm1": (2830, 2695), "text": "Aldehyde C–H doublet support"}],
        negative=[],
        examples=[1730.0],
    ),
    _entry(
        id="carbonyl_ketone",
        range_cm1=(1725, 1705),
        label="C=O stretch (ketone)",
        typical_shape=["sharp"],
        typical_intensity=["strong"],
        notes="Simple saturated ketones near ~1715; conjugation lowers toward ~1685.",
        positive=[],
        negative=[],
        examples=[1715.0],
    ),
    _entry(
        id="carbonyl_amide_i",
        range_cm1=(1690, 1630),
        label="Amide I (C=O stretch)",
        typical_shape=["medium"],
        typical_intensity=["strong", "variable"],
        notes="Amide I overlaps C=C; often with amide II (1650–1580) and N–H stretch (~3500–3300).",
        positive=[{"range_cm1": (1650, 1580), "text": "Amide II support"}, {"range_cm1": (3500, 3300), "text": "N–H stretch support"}],
        negative=[],
        examples=[1650.0],
    ),
    _entry(
        id="carbonyl_anhydride_asym",
        range_cm1=(1820, 1760),
        label="C=O stretch (anhydride, asym)",
        typical_shape=["sharp"],
        typical_intensity=["strong"],
        notes="Anhydrides often show two strong C=O bands: ~1820 and ~1760.",
        positive=[{"range_cm1": (1760, 1700), "text": "Second anhydride C=O band support"}],
        negative=[],
        examples=[1810.0],
    ),
    _entry(
        id="carbonyl_anhydride_sym",
        range_cm1=(1760, 1700),
        label="C=O stretch (anhydride, sym)",
        typical_shape=["sharp"],
        typical_intensity=["strong"],
        notes="Pairs with anhydride asym band at higher wavenumber.",
        positive=[{"range_cm1": (1820, 1760), "text": "Anhydride asym band support"}],
        negative=[],
        examples=[1760.0],
    ),
    _entry(
        id="carbonyl_acid_halide",
        range_cm1=(1815, 1770),
        label="C=O stretch (acid halide)",
        typical_shape=["sharp"],
        typical_intensity=["strong"],
        notes="Acid chlorides high frequency; very reactive; may overlap anhydrides.",
        positive=[],
        negative=[{"range_cm1": (3300, 2500), "text": "Acid O–H would argue against acid halide"}],
        examples=[1800.0],
    ),

    # --- C=C and ring modes ---
    _entry(
        id="alkene_cc",
        range_cm1=(1680, 1620),
        label="C=C stretch (alkene)",
        typical_shape=["medium"],
        typical_intensity=["weak", "medium", "variable"],
        notes="Often weak; may overlap amide I/II and aromatic bands.",
        positive=[{"range_cm1": (3100, 3000), "text": "sp2 C–H stretch supports alkene/aromatic"}],
        negative=[],
        examples=[1640.0],
    ),
    _entry(
        id="aromatic_cc_1600",
        range_cm1=(1620, 1585),
        label="Aromatic C=C stretch (~1600)",
        typical_shape=["medium"],
        typical_intensity=["weak", "medium", "variable"],
        notes="Usually paired with ~1500 cm^-1 ring band; overtones in 2000–1600 may appear.",
        positive=[{"range_cm1": (1515, 1450), "text": "Aromatic ~1500 band support"}, {"range_cm1": (2000, 1600), "text": "Aromatic overtones support"}],
        negative=[],
        examples=[1600.0],
    ),
    _entry(
        id="aromatic_cc_1500",
        range_cm1=(1515, 1450),
        label="Aromatic C=C stretch (~1500)",
        typical_shape=["medium"],
        typical_intensity=["weak", "medium", "variable"],
        notes="Common aromatic ring mode; paired with ~1600 cm^-1.",
        positive=[{"range_cm1": (1620, 1585), "text": "Aromatic ~1600 band support"}],
        negative=[],
        examples=[1500.0],
    ),

    # --- Nitro ---
    _entry(
        id="nitro_no2_asym",
        range_cm1=(1660, 1500),
        label="NO2 asym stretch (nitro)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong"],
        notes="Usually strong; pairs with symmetric NO2 stretch ~1390–1260.",
        positive=[{"range_cm1": (1390, 1260), "text": "NO2 symmetric stretch support"}],
        negative=[],
        examples=[1550.0],
    ),
    _entry(
        id="nitro_no2_sym",
        range_cm1=(1390, 1260),
        label="NO2 sym stretch (nitro)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong"],
        notes="Pairs with asymmetric NO2 stretch ~1660–1500.",
        positive=[{"range_cm1": (1660, 1500), "text": "NO2 asymmetric stretch support"}],
        negative=[],
        examples=[1360.0],
    ),

    # --- C–O region ---
    _entry(
        id="co_stretch_ethers_alcohols_esters",
        range_cm1=(1260, 1000),
        label="C–O stretch (ethers/esters/alcohols)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong", "variable"],
        notes="Fingerprint region; multiple bands common; assignment requires context.",
        positive=[{"range_cm1": (1760, 1680), "text": "Carbonyl nearby supports ester/acid derivatives"}, {"range_cm1": (3600, 3200), "text": "O–H stretch supports alcohol"}],
        negative=[],
        examples=[1050.0, 1150.0],
    ),

    # --- Sulfur / phosphorus / silicon ---
    _entry(
        id="sulfoxide_so",
        range_cm1=(1070, 1030),
        label="S=O stretch (sulfoxide)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong"],
        notes="Often strong; overlaps C–O region.",
        positive=[],
        negative=[],
        examples=[1050.0],
    ),
    _entry(
        id="sulfone_so2_asym",
        range_cm1=(1350, 1290),
        label="SO2 asym stretch (sulfone)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong"],
        notes="Typically strong; pairs with symmetric SO2 stretch ~1170–1120.",
        positive=[{"range_cm1": (1170, 1120), "text": "SO2 symmetric stretch support"}],
        negative=[],
        examples=[1310.0],
    ),
    _entry(
        id="sulfone_so2_sym",
        range_cm1=(1170, 1120),
        label="SO2 sym stretch (sulfone)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong"],
        notes="Pairs with asymmetric SO2 stretch ~1350–1290.",
        positive=[{"range_cm1": (1350, 1290), "text": "SO2 asymmetric stretch support"}],
        negative=[],
        examples=[1140.0],
    ),
    _entry(
        id="phosphate_po",
        range_cm1=(1260, 1150),
        label="P=O stretch (phosphate/phosphoryl)",
        typical_shape=["sharp", "medium"],
        typical_intensity=["strong", "variable"],
        notes="Often strong; overlaps C–O region; confirm with P–O bands 1100–900.",
        positive=[{"range_cm1": (1100, 900), "text": "P–O stretch region support"}],
        negative=[],
        examples=[1230.0],
    ),
    _entry(
        id="phosphate_po_stretch_region",
        range_cm1=(1100, 900),
        label="P–O stretch (phosphate)",
        typical_shape=["medium", "broad"],
        typical_intensity=["strong", "variable"],
        notes="Often multiple bands; overlaps Si–O and C–O fingerprint region.",
        positive=[{"range_cm1": (1260, 1150), "text": "P=O support"}],
        negative=[],
        examples=[1050.0, 980.0],
    ),
    _entry(
        id="sio_si_o",
        range_cm1=(1130, 1000),
        label="Si–O stretch (siloxane/silicate)",
        typical_shape=["broad", "medium"],
        typical_intensity=["strong", "variable"],
        notes="Often broad/strong; overlaps C–O region.",
        positive=[],
        negative=[],
        examples=[1100.0],
    ),

    # --- Aromatic C–H out-of-plane ---
    _entry(
        id="aromatic_ch_out_of_plane",
        range_cm1=(900, 675),
        label="Aromatic C–H out-of-plane bend",
        typical_shape=["sharp", "medium"],
        typical_intensity=["medium", "strong", "variable"],
        notes="Substitution pattern dependent; supports aromatic ring with 1600/1500 bands.",
        positive=[{"range_cm1": (1620, 1450), "text": "Aromatic ring bands support"}],
        negative=[],
        examples=[750.0, 830.0],
    ),
]


def get_library_v2() -> List[Dict[str, Any]]:
    """Return a shallow copy of library v2 entries."""

    return list(FTIR_LIBRARY_V2)
