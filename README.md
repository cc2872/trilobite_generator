# The trilobite generator: parameters, measurement, and purpose

*schema 6.1, instrument 2.1*
*22 September 2026*
*by Claire Choi the Trilobiter*

## How to read this

An animal is a dictionary of 98 numbers. Most are ratios of some other dimension so that the animal scales (sanely) sensibly. A few are millimetres or degrees for absolute measurements. They are grouped as the whole-body frame, then a three-by-three grid of head, thorax and tail against shape, surface and margin, then the joint. Each entry gives the key, a one-line explanation, and the default with its range in brackets. Parameters that can change the enrolment reading are listed under "Closure chain" at the end. The rest are cosmetic for the instrument.

## Frame: the whole animal (12)

- **`length`**: Whole animal, head tip to tail shield tip (spines extra). Default 130 [60–250] mm.
- **`width`**: Widest point of the body. Default 65 [30–140] mm.
- **`relief`**: Dorsal height of the vault at the axis. Default 16 [6–40] mm.
- **`segCount`**: Number of articulating thoracic segments. Default 6 [2–16].
- **`cephFrac`**: Cephalon length / body length. Default 0.33 [0.18–0.45].
- **`pygFrac`**: Pygidium length / body length (spines extra). Default 0.14 [0.05–0.4].
- **`wall`**: Exoskeleton thickness (keep ≥ 1.5 for FDM). Default 2.0 [1.2–4.0] mm.
- **`widthMaxPos`**: Position of max width along the body (0 head, 1 tail). Default 0.3 [0.1–0.8].
- **`widthHeadFront`**: Width at the head's front third / max width. Default 0.85 [0.3–1.0].
- **`widthThoraxFront`**: Width of segment 0 / max width (the head–thorax shoulder). Default 0.92 [0.6–1.0].
- **`widthThoraxRear`**: Width at the last segment / max width. Default 0.62 [0.3–1.0].
- **`widthTail`**: Width near the tail tip / max width. Default 0.25 [0.0–0.8].

## Glabella and occipital ring: the raised middle of the head (11)

- **`glabInflate`**: Glabella front width / rear width (club shape). Default 1.25 [0.6–2.0].
- **`glabRise`**: Glabella height above the cheeks / relief. Default 0.18 [0.0–0.6].
- **`glabLobes`**: Pairs of lateral glabellar furrows. Default 3 [0–4].
- **`glabFront`**: How the glabella closes at the front: 2.5 ovoid nose, 6+ blunt. Default 2.5 [1.5–8.0].
- **`headDomeExp`**: Superellipsoid exponent of the head vault (1.5 = tent-like, 2 = ellipsoid, higher = flatter top). Default 1.5 [1.2–4.0].
- **`headDomeFill`**: Dome semi-axes / head half-width and length (the rest is flat border). Default 0.82 [0.5–1.0].
- **`headOutlineExp`**: Superellipse exponent of the head front (2 = ellipse, higher = squarer). Default 2.15 [1.2–5.0].
- **`cephParallel`**: Fraction of head length that is parallel-sided. Default 0.45 [0.0–0.8].
- **`headRelief`**: Head apex height / relief (the occipital ring always holds the hinge height). Default 1.0 [0.6–1.6].
- **`headWall`**: Head shell thickness / wall (segment 0's stepped front follows it). Default 1.0 [0.6–2.5].
- **`occipitalSpine`**: Spine on the occipital ring / head length (0 = none). Default 0.0 [0.0–1.5].

## Cheek, eye and genal angle: the sides of the head (32)

- **`eyeSize`**: Eye radius / head half-width (0 = blind). Default 0.14 [0.0–0.45].
- **`genalSpine`**: Genal spine length / head length (0 = none). Default 0.35 [0.0–1.5].
- **`headRearArc`**: Crescent: the rear edge bows back from the axis to the genal angles by this fraction of head length (0 = straight rear, v4). Default 0.0 [0.0–0.7].
- **`eyePos`**: Along the head, 0 rear → 1 front. Default 0.45 [0.1–0.9].
- **`eyeArc`**: Angular extent of the visual surface. Default 150 [60–300] deg.
- **`eyeHeight`**: Dome height of the eye / eye radius. Default 0.8 [0.2–1.6].
- **`eyeLat`**: Eye centre / head half-width (0 = hug the glabella, the v4 rule). Phacops scan: 0.55. Default 0.0 [0.0–0.9].
- **`eyeProfile`**: Super-Gaussian exponent of the eye dome: 2 = round, 8 = flat-topped drum (schizochroal). Default 4.0 [2.0–8.0].
- **`eyeSolid`**: 1 = revolved eye primitive with lens lattice (`eye_solid.py`); 0 = the v4 heightfield bump. Default 0 [0–1].
- **`eyeSlope`**: Visual band from vertical: Erbenochile 0, Phacops 15, Isotelus ~45. Default 15.0 [0.0–50.0] deg.
- **`eyeShade`**: Palpebral brim past the band top / R (Erbenochile ~0.15). Default 0.0 [0.0–0.3].
- **`lensD`**: Lens diameter / R (holochroal ~0.05, schizochroal 0.1–0.2). Default 0.16 [0.02–0.4].
- **`lensGap`**: Sclera between lenses / lens diameter (0 = holochroal). Default 0.3 [0.0–0.6].
- **`lensRise`**: Lens cap height / lens radius. Default 0.35 [0.0–1.0].
- **`eyeStalk`**: Pedunculate eye: stalk length / eye radius (0 = sessile, sitting on the cheek). Default 0.0 [0.0–4.0].
- **`eyeStalkR`**: Stalk radius / eye radius. Default 0.45 [0.15–1.0].
- **`eyeLean`**: A stalked eye tips forward from vertical; 90° looks straight ahead over the front margin. Default 0 [0–90] deg.
- **`borderWidth`**: Raised border / head half-width (0 = none). Default 0.1 [0.0–0.3].
- **`genalSweep`**: How far the cheeks sweep back along the shoulder / segment pitch. Default 0.8 [0.0–2.5].
- **`genalCurve`**: Inward curl of the genal spines. Default 20 [-60–60] deg.
- **`genalPath`**: Sideways offset of the arm's centreline as a formula in s (0 = root, 1 = tip), scaled by genalCurve. e.g. `s**2` flares late, `s` straight, `sin(pi*s/2)` flares early, `-0.5*s+s**2` hugs the thorax then flares. Default `s**2` [0–0].
- **`genalWidth`**: Thickness of the crescent's arm across, / head half-width (harpetid 0.3). Default 0.12 [0.03–0.5].
- **`genalWidthMM`**: Absolute arm thickness in mm; 0 = use genalWidth (fraction of half-width). Default 0.0 [0.0–30.0].
- **`genalTaper`**: End of the arm: 0.5 = blunt/rounded, 3 = drawn to a point. Default 2.0 [0.5–3.0].
- **`headRearExp`**: 1 = the rear edge leaves the axis at once (thin band); higher = it stays straight and bows late (thick band, short horns). Default 2.2 [1.0–5.0].
- **`headProngs`**: Tines of the anterior prong at the head front (0 = none, 3 = Walliserops-style trident). Default 0 [0–5].
- **`headProngLen`**: Anterior prong length / head length. Default 0.6 [0.1–2.0].
- **`headProngSplay`**: Total fan angle of the anterior tines in the plan. Default 30 [0–90] deg.
- **`headProngStem`**: Shared shaft before the tines split, fraction of prong length (Walliserops ~0.5). Default 0.0 [0.0–0.8].
- **`headProngCenter`**: Middle tine length / outer tines (odd tine counts). Default 1.0 [0.5–2.0].
- **`headProngWidth`**: Root radius / margin height. Default 0.45 [0.2–1.2].
- **`headProngCurl`**: Tines lift (+) or dip (-) at the split. Default 0.0 [-40–40] deg.

## Axial rings: the ring chain over the hinges (8)

- **`axisFrac`**: Axial lobe width / local body width. Default 0.33 [0.2–0.5].
- **`axisRise`**: Ring stands this fraction of relief above the vault. Default 0.15 [0.0–0.4].
- **`furrowDepth`**: Depth of the furrows at full expression. Default 1.2 [0.0–3.0] mm.
- **`effacement`**: 0 = fully sculpted, 1 = smooth (Nileus). Default 0.0 [0.0–1.0].
- **`axisSigma`**: Gaussian half-width of the axial dome / half-width (tent vault). Default 0.55 [0.25–1.0].
- **`ringArch`**: Convexity of each axial ring between its two hinges / relief. Default 0.22 [0.0–0.5].
- **`tent`**: 0 = fulcrum vault (v4), 1 = axial dome + straight pleural slope (reference sculpt). Default 1.0 [0.0–1.0].
- **`axialSpine`**: Dorsal spine on every thoracic ring / relief (0 = none). Default 0.0 [0.0–2.5].

## Pleurae: the ribs, an inner run to the fulcrum then a blade (13)

- **`overlap`**: Rear flap length as a fraction of segment pitch. Default 0.5 [0.1–0.8].
- **`fulcrum`**: Where the pleura bends down, as a fraction of half-width. Default 0.55 [0.3–0.9].
- **`pleuralSlope`**: Linear drop of the pleural slope over the half-width / relief (tent vault). Default 0.74 [0.2–1.2].
- **`marginHeight`**: Pleural margin height / relief. Default 0.34 [0.05–0.6].
- **`bladeCamber`**: Convexity of each pleural blade along its chord / relief. Default 0.14 [0.0–0.4].
- **`bladeChord`**: Fore-aft width of the pleural blade beyond its root / segment pitch (1.3 = the v4 full plate, 0.9 = separate ribs). Default 1.3 [0.5–1.3].
- **`tipSweep`**: How far the pleural blades sweep back / pitch. Default 0.5 [0.0–2.0].
- **`tipTaper`**: How much the blade narrows toward its tip (0 = square, 0.95 = needle). Default 0.55 [0.0–0.95].
- **`spineBase`**: Extra needle spine beyond the blade tip, segment 0 / half-width. Default 0.0 [0.0–1.2].
- **`spineGrad`**: Change in spine length from first to last segment. Default 0.0 [-1.0–1.0].
- **`spineSweep`**: Pleural spines sweep back by this angle. Default 50 [0–80] deg.
- **`macroIndex`**: Index of a segment with extra-long spines (-1 = none). Default -1 [-1–15].
- **`macroAmp`**: Extra spine length on that segment / half-width. Default 0.8 [0.0–2.0].

## Tail axis (5)

- **`pygRings`**: Axial rings on the pygidium. Default 4 [0–12].
- **`tailRelief`**: Tail apex height / relief (the axis always holds the hinge height over the front joint). Default 1.0 [0.5–1.4].
- **`tailDomeExp`**: Fall-off of the tail along the body: 2 = elliptical, higher = flat top with a steep rear. Default 2.0 [1.2–5.0].
- **`tailWall`**: Tail shell thickness / wall. Default 1.0 [0.6–2.5].
- **`termSpine`**: Single median terminal spine / shield length (0 = none). Default 0.0 [0.0–2.5].

## Tail field and margin (12)

- **`pygWidth`**: Tail shield width / last segment width. Default 0.9 [0.5–1.1].
- **`pygSpine`**: Paired tail spines / shield length (0 = none). Default 0.9 [0.0–2.0].
- **`pygMarginal`**: Spines around the tail margin (0 = none). Default 0 [0–10].
- **`pygSplay`**: Outward angle of the tail spines. Default 20 [0–45] deg.
- **`pygMarginalLen`**: Marginal spine length / shield length. Default 0.5 [0.1–1.5].
- **`tailProngs`**: Tines of the posterior prong at the tail tip (0 = none; the paired forks are pygSpine, the margin itself). Default 0 [0–5].
- **`tailProngLen`**: Posterior prong length / tail length. Default 0.5 [0.1–2.0].
- **`tailProngSplay`**: Total fan angle of the posterior tines in the plan. Default 30 [0–90] deg.
- **`tailProngStem`**: Shared shaft before the tines split, fraction of prong length. Default 0.0 [0.0–0.8].
- **`tailProngCenter`**: Middle tine length / outer tines (odd tine counts). Default 1.0 [0.5–2.0].
- **`tailProngWidth`**: Root radius / margin height. Default 0.45 [0.2–1.2].
- **`tailProngCurl`**: Tines lift (+) or dip (-) at the split. Default 0.0 [-40–40] deg.

## Joint and print: manufacturing, not morphology (5)

- **`maxAngle`**: Ventral flexion per joint before the stop engages. Default 18 [4–40] deg.
- **`clearance`**: Gap between moving parts. Default 0.3 [0.15–0.6] mm.
- **`nKnuckles`**: Odd number across the hinge. Default 3 [3–7].
- **`boreDia`**: For 1.75 mm filament pins. Default 1.95 [1.6–3.0] mm.
- **`barrelR`**: . Default 2.6 [1.8–4.0] mm.

## Closure chain

The parameters that can move the reading, because they shape the parts the instrument collides or the head and tail edges it tests for closure. Every other parameter is outside the chain.

`length`, `width`, `relief`, `segCount`, `cephFrac`, `pygFrac`, `marginHeight`, `fulcrum`, `overlap`, `wall`, `widthMaxPos`, `widthHeadFront`, `widthThoraxFront`, `widthThoraxRear`, `widthTail`, `axisFrac`, `axisRise`, `tent`, `axisSigma`, `pleuralSlope`, `ringArch`, `bladeCamber`, `headOutlineExp`, `headDomeExp`, `headDomeFill`, `headRelief`, `tailRelief`, `tailDomeExp`, `cephParallel`, `glabRise`, `genalSpine`, `headRearArc`, `tipSweep`, `bladeChord`, `tipTaper`, `spineBase`, `spineGrad`, `macroIndex`, `macroAmp`, `spineSweep`, `pygWidth`, `pygSpine`, `axialSpine`, `clearance`.

## The measurement

Give the instrument a dictionary and it returns a record. The fields that matter:

- **`theta_joint_deg`**: The angle each joint had reached when the sweep stopped, to 0.1 degrees. This is the reading. Typical values 20 to 35.
- **`total_deg`**: That angle times the number of joints (segments plus one). Roughly how far around the animal got; real closure happens before 360 because the tail meets the head.
- **`limited_by`**: Why it stopped. `closed`: the tail reached the head, the animal enrolled. `anatomy`: two shell parts collided first, it jammed. `bound`: it ran out of bevel before either, the reading is censored, not a result. `invalid`: it could not be measured (a part built in pieces, or the resting pose interfered).
- **`stopped_by`**: For `anatomy`, which pair of parts touched. This is the diagnostic that says what about the shape jammed it.
- **`closure_gap_mm`**: How far the tail's edge was from the head's at the stop.
- **`s_tail`**: Where the tail landed relative to the head: negative means over the head, near zero means edge to edge, positive means under it.
- **`enroll_class`**: The named style, from pre-registered rules on the numbers above: sphaeroidal (edge to edge), double (tail under head), spiral (tail over head), discoidal (a short flat curl with real overlap), or open (never closed).
- **`bevel_built_deg`**: Which of the 45 / 35 / 25 bevel ladder the animal was built at; anything below 45 means a part would not build whole at the full bevel.
- **`unsane_parts`**: Any part that came out as more than one body. Non-empty means the geometry, not the animal, failed.

The constants (step size, closure tolerance, the discoidal thresholds) are written into every record, so a result carries its own ruler.

Two things about the number that people misread. First, theta is per joint and uniform: the instrument assumes every joint bends the same amount. Real animals could bend one joint more than another, so the reading is a lower bound on what the animal could do. Second, the stop is shell on shell; there are no soft-tissue limits, no interlocking structures, no muscle. Both are stated in the paper; both mean the number is conservative in a known direction.
