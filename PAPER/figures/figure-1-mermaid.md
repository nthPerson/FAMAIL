# Figure 1 — Mermaid source for the four variants

Companion to [`figure-1.md`](figure-1.md). One fenced Mermaid block per variant, written for
**import into diagramming tools** (draw.io: Arrange → Insert → Advanced → Mermaid; Excalidraw:
"Mermaid to Excalidraw"; quick preview: mermaid.live), then refined by hand.

**What Mermaid can and cannot carry here.** Mermaid is a graph language: it encodes each
variant's *structure* — panels, regions, edits, flows, containment, styling intent — as boxes
and edges. It cannot draw literal city grids, scatter symbols, smooth heat fields, or curved
tapered polylines. Those are the post-import refinements; each block's `%%` comments say
exactly what to redraw. Symbol conventions ride inside labels (□ = taxi presence, ✕ = pickup)
per `figure-1.md`.

**House constraints carried in the code:** grayscale-safe (accent items are *also* dash-bordered,
so meaning survives mono print); exactly one accent (#2166ac blue) via the `accent`/`accentFill`
classes; no numbers, no "pillar", no tool names in any label.

**Portability notes:** `~~~` is Mermaid's invisible edge (layout ordering only) — if a tool
rejects it, replace with `---` and delete the line after import. All labels are quoted; `<br/>`
is the line break.

---

## Variant A — "Three panels, one city" (safest; mirrors the argument order)

```mermaid
flowchart LR
  %% VARIANT A: three identical stylized city grids, left->right.
  %% Post-import: redraw each panel as a ~10x8 cell grid (thin light gridlines);
  %% place over-served upper-left, under-served lower-right; keep panels identical
  %% except for the edits each one adds. Panel titles above each grid.

  subgraph P1["Panel 1 — the service gap"]
    direction TB
    P1OS["over-served<br/>□ □ □ □ □<br/>□ ✕ □ □ □<br/>□ □ □ □ □"]:::neutral
    P1US["under-served<br/>✕ ✕ ✕ ✕ ✕<br/>✕ ✕ □ ✕ ✕<br/>✕ ✕ ✕ ✕ ✕"]:::neutral
    P1LEG["□ = taxi presence&nbsp;&nbsp;✕ = pickup"]:::legend
    P1OS ~~~ P1US
    P1US ~~~ P1LEG
  end

  subgraph P2["Panel 2 — trim: relocate excess pickups"]
    direction TB
    P2G["✕ ✕ ghosted originals<br/>(inside hotspot)"]:::ghost
    P2N["✕ ✕ relocated pickups<br/>(just outside hotspot)"]:::neutral
    P2US["under-served —<br/>pixel-identical to Panel 1"]:::neutral
    P2A["levels down:<br/>under-served untouched"]:::annot
    P2G -->|"trim move, two cells max"| P2N
    P2N ~~~ P2US
    P2US ~~~ P2A
  end

  subgraph P3["Panel 3 — lift: reroute seeking tails"]
    direction TB
    P3IN["trajectory enters<br/>(solid thin polyline)"]:::neutral
    P3ANC["anchor state<br/>(never moves)"]:::neutral
    P3TAIL["last four seeking states<br/>gentle tapered bend"]:::accent
    subgraph P3GLOW["value of added presence (supply gradient)"]
      direction TB
      P3X["✕ pickup lands<br/>inside the glow"]:::accent
    end
    P3IN --> P3ANC
    P3ANC -.->|"detour, dashed accent"| P3TAIL
    P3TAIL -.-> P3X
  end

  P1 --> P2
  P2 --> P3

  style P3GLOW fill:#dbe9f6,stroke:#2166ac,color:#0b3d66
  classDef neutral fill:#f7f7f7,stroke:#555555,color:#111111
  classDef ghost fill:#ffffff,stroke:#aaaaaa,color:#888888,stroke-dasharray:3 3
  classDef accent fill:#ffffff,stroke:#2166ac,color:#0b3d66,stroke-dasharray:5 3
  classDef annot fill:#ffffff,stroke:#dddddd,color:#666666
  classDef legend fill:#ffffff,stroke:#cccccc,color:#333333
  %% Edge styling (indices = definition order; recount if you add edges):
  %% edge 4 (P3ANC -.-> P3TAIL) and edge 5 (P3TAIL -.-> P3X) are the lift detour.
  linkStyle 4 stroke:#2166ac
  linkStyle 5 stroke:#2166ac
```

*Post-import checklist:* panel grids identical; the glow is a soft single-hue tint (~12%
opacity) under the under-served region; the tail bend is a smooth curve, not a right angle;
under-served ✕ marks stay dense (starved of squares, not empty).

---

## Variant B — "Before/after split city" (single panel, maximal gestalt)

```mermaid
flowchart LR
  %% VARIANT B: one wide panel, same city twice, thin vertical divide between halves,
  %% one large arrow crossing the divide. Both edits appear simultaneously on the right.
  %% Post-import: draw both halves as the same grid; the divide is a thin vertical rule.

  subgraph BEF["before"]
    direction TB
    BOS["over-served hotspot<br/>□ dense, ✕ few"]:::neutral
    BUS["under-served<br/>✕ dense, □ one"]:::neutral
    BLEG["□ = taxi presence&nbsp;&nbsp;✕ = pickup"]:::legend
    BOS ~~~ BUS
    BUS ~~~ BLEG
  end

  BEF ==>|"edit: trim + lift"| AFT

  subgraph AFT["after"]
    direction TB
    AOS["hotspot: ✕ ✕ moved out<br/>(two short trim arrows)"]:::neutral
    ATAIL["one seeking tail<br/>bends in (dashed accent)"]:::accent
    subgraph AGLOW["under-served — now glowing"]
      direction TB
      ANEW["□ gained presence (accent outline)<br/>✕ landed pickup"]:::accent
    end
    AOS ~~~ ATAIL
    ATAIL -.-> ANEW
  end

  CO1["trim<br/>(demand moves)"]:::annot
  CO2["lift<br/>(supply moves with the tail)"]:::annot
  CO1 --- AOS
  CO2 --- ATAIL

  style AGLOW fill:#dbe9f6,stroke:#2166ac,color:#0b3d66
  classDef neutral fill:#f7f7f7,stroke:#555555,color:#111111
  classDef accent fill:#ffffff,stroke:#2166ac,color:#0b3d66,stroke-dasharray:5 3
  classDef annot fill:#ffffff,stroke:#dddddd,color:#666666
  classDef legend fill:#ffffff,stroke:#cccccc,color:#333333
  %% edge 1 (BEF ==> AFT) is the big divide-crossing arrow; keep it bold after import.
  %% edge 2 (ATAIL -.-> ANEW) is the lift detour.
  linkStyle 2 stroke:#2166ac
  %% CO1/CO2 are callouts: after import, convert their edges to thin leader lines.
```

*Post-import checklist:* the two callouts become tiny labels with leader lines; the gained
□ in the after-city is drawn in accent outline only (fill stays white); this variant shows
no isolated trim failure mode — that is by design (see `figure-1.md`).

---

## Variant C — "Mechanism ribbon" (pipeline; the only variant with the downstream recipe)

```mermaid
flowchart LR
  %% VARIANT C: four-stage ribbon, each stage a small box with a micro-diagram.
  %% The two cross-edges out of Attribution are the point of this variant:
  %% they draw the paper's two-mechanism contribution explicitly.

  subgraph S1["real corpus 𝒯"]
    direction TB
    C1S["stack of thin polylines<br/>(trajectories)"]:::neutral
    C1H["one highlighted"]:::accent
    C1S ~~~ C1H
  end

  subgraph S2["attribution"]
    direction TB
    C2D["deficit map<br/>where unfairness lives<br/>(hot cells: over-served cluster)"]:::neutral
    C2V["value-of-presence map<br/>fairness gain per unit of added supply<br/>(hot cells: under-served region)"]:::accentFill
    C2D ~~~ C2V
  end

  subgraph S3["bounded edits"]
    direction TB
    C3T["trim arrow<br/>two cells max"]:::neutral
    C3L["tapered tail bend"]:::accent
    C3B["dashed square around each edit<br/>= the two-cell ball"]:::ghost
    C3T ~~~ C3L
    C3L ~~~ C3B
  end

  subgraph S4["upweighted imitation"]
    direction TB
    C4E["edited slice drawn thicker<br/>(weight w on the edited slice)"]:::accent
    C4R["rest of the corpus<br/>(thin, weight one)"]:::neutral
    C4P["policy"]:::neutral
    C4E --> C4P
    C4R --> C4P
  end

  S1 --> S2
  S2 --> S3
  S3 --> S4
  C2D -->|"drives trim"| C3T
  C2V -.->|"drives lift"| C3L

  classDef neutral fill:#f7f7f7,stroke:#555555,color:#111111
  classDef ghost fill:#ffffff,stroke:#aaaaaa,color:#888888,stroke-dasharray:3 3
  classDef accent fill:#ffffff,stroke:#2166ac,color:#0b3d66,stroke-dasharray:5 3
  classDef accentFill fill:#dbe9f6,stroke:#2166ac,color:#0b3d66
  %% edge order: C4E->C4P, C4R->C4P, S1->S2, S2->S3, S3->S4, C2D->C3T, C2V-.->C3L
  linkStyle 6 stroke:#2166ac
```

*Post-import checklist:* the two attribution boxes become micro heat-maps (tiny grids with a
few dark cells in the right regions); the "drives trim"/"drives lift" cross-edges should
visually bypass the stage arrows (route them below the ribbon); C4E/C4R read best as a
re-drawn trajectory stack with two line weights. The ∂-notation was spelled out in words in
the label to keep it self-explanatory — restore "∂fairness/∂supply" after import if preferred.

---

## Variant D — "A + gradient inset" (recommended production target)

```mermaid
flowchart LR
  %% VARIANT D: Variant A's three panels with two refinements:
  %% (1) Panel 3's glow is a smooth single-hue FIELD with faint iso-contours
  %%     (represented here as nested containment: outer/mid/peak);
  %% (2) Panels 2 and 3 carry paired schematic insets encoding the ablation story.
  %% Insets stay schematic — NO axis numbers, or they will be read as data.

  subgraph P1["Panel 1 — the service gap"]
    direction TB
    D1OS["over-served<br/>□ □ □ □ □<br/>□ ✕ □ □ □<br/>□ □ □ □ □"]:::neutral
    D1US["under-served<br/>✕ ✕ ✕ ✕ ✕<br/>✕ ✕ □ ✕ ✕<br/>✕ ✕ ✕ ✕ ✕"]:::neutral
    D1LEG["□ = taxi presence&nbsp;&nbsp;✕ = pickup"]:::legend
    D1OS ~~~ D1US
    D1US ~~~ D1LEG
  end

  subgraph P2["Panel 2 — trim: relocate excess pickups"]
    direction TB
    D2G["✕ ✕ ghosted originals"]:::ghost
    D2N["✕ ✕ relocated just outside"]:::neutral
    D2US["under-served — untouched"]:::neutral
    D2I["inset: over-served ↓ · under-served —"]:::annot
    D2G -->|"trim move, two cells max"| D2N
    D2N ~~~ D2US
    D2US ~~~ D2I
  end

  subgraph P3["Panel 3 — lift: reroute seeking tails"]
    direction TB
    D3IN["trajectory enters"]:::neutral
    D3ANC["anchor state (never moves)"]:::neutral
    D3TAIL["last four seeking states<br/>gentle tapered bend"]:::accent
    subgraph D3F1["gradient field — outer contour (light)"]
      direction TB
      subgraph D3F2["mid contour"]
        direction TB
        D3PK["peak (darkest)<br/>✕ pickup lands here"]:::accent
      end
    end
    D3I["inset: over-served ↓ · under-served ↑"]:::annotAccent
    D3IN --> D3ANC
    D3ANC -.->|"detour, dashed accent"| D3TAIL
    D3TAIL -.-> D3PK
    D3F1 ~~~ D3I
  end

  P1 --> P2
  P2 --> P3

  style D3F1 fill:#eaf2fa,stroke:#9dc0e0,color:#0b3d66
  style D3F2 fill:#c9dff2,stroke:#5b93c4,color:#0b3d66
  classDef neutral fill:#f7f7f7,stroke:#555555,color:#111111
  classDef ghost fill:#ffffff,stroke:#aaaaaa,color:#888888,stroke-dasharray:3 3
  classDef accent fill:#ffffff,stroke:#2166ac,color:#0b3d66,stroke-dasharray:5 3
  classDef annot fill:#ffffff,stroke:#dddddd,color:#666666
  classDef annotAccent fill:#ffffff,stroke:#2166ac,color:#0b3d66
  %% edge order: D2G->D2N, D3IN->D3ANC, D3ANC-.->D3TAIL, D3TAIL-.->D3PK, P1->P2, P2->P3
  linkStyle 2 stroke:#2166ac
  linkStyle 3 stroke:#2166ac
```

*Post-import checklist:* the nested contour boxes become 2–3 faint iso-lines over a smooth
light→dark single-hue field (intensity = supply-gradient value); the paired insets become
miniature two-bar glyphs in matching corners of Panels 2 and 3 (the pairing IS the message:
Panel 2's "under-served —" is answered by Panel 3's "under-served ↑"); keep both insets
axis-less and number-free. Note the nested-field colors are two tints of the SAME blue —
still one accent hue.
