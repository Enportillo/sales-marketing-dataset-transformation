"""Helpers de visualizacion para evitar solapamiento de labels en graficos Plotly."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def ensure_min_margin(fig, min_top: int | None = None, min_bottom: int | None = None):
    """Asegura margenes minimos sin sobrescribir otros valores ya definidos."""
    current = fig.layout.margin
    t = int(current.t) if current and current.t is not None else 0
    b = int(current.b) if current and current.b is not None else 0
    l = int(current.l) if current and current.l is not None else None
    r = int(current.r) if current and current.r is not None else None

    if min_top is not None:
        t = max(t, int(min_top))
    if min_bottom is not None:
        b = max(b, int(min_bottom))

    margin_kwargs = {"t": t, "b": b}
    if l is not None:
        margin_kwargs["l"] = l
    if r is not None:
        margin_kwargs["r"] = r
    fig.update_layout(margin=margin_kwargs)


def apply_outside_text_anti_overlap(fig, min_top: int = 70):
    """Mejora legibilidad para labels fuera de barras y reduce solapamientos."""
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    ensure_min_margin(fig, min_top=min_top)


def add_non_overlapping_vline_labels(
    fig,
    lines: Iterable[dict],
    data_min: float,
    data_max: float,
    closeness_ratio: float = 0.03,
    base_y: float = 1.08,
):
    """Dibuja lineas verticales con etiquetas separadas cuando los valores son cercanos."""
    items = [dict(item) for item in lines]
    if not items:
        return

    data_range = max(float(data_max) - float(data_min), 1e-9)
    closeness = closeness_ratio * data_range

    # Agrega lineas base
    for item in items:
        fig.add_vline(
            x=float(item["x"]),
            line_dash=item.get("line_dash", "dash"),
            line_color=item.get("line_color", "#111111"),
        )

    # Agrupa lineas cercanas para separar etiquetas
    sorted_idx = sorted(range(len(items)), key=lambda i: float(items[i]["x"]))
    groups: list[list[int]] = []
    for idx in sorted_idx:
        if not groups:
            groups.append([idx])
            continue
        last_idx = groups[-1][-1]
        if abs(float(items[idx]["x"]) - float(items[last_idx]["x"])) <= closeness:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    max_y = base_y
    for group in groups:
        n = len(group)
        if n == 1:
            y_vals = [base_y]
            shifts = [0]
        else:
            # Distribuye etiquetas en X y Y para maximizar legibilidad
            y_vals = list(np.linspace(base_y + 0.06, base_y - 0.06, n))
            shifts = [int(v) for v in np.linspace(-28, 28, n)]

        for local_i, idx in enumerate(group):
            item = items[idx]
            y_val = float(y_vals[local_i])
            max_y = max(max_y, y_val)
            fig.add_annotation(
                x=float(item["x"]),
                y=y_val,
                xref="x",
                yref="paper",
                text=str(item.get("text", "")),
                showarrow=False,
                xshift=shifts[local_i],
                font=dict(
                    color=item.get("font_color", item.get("line_color", "#111111")),
                    size=int(item.get("font_size", 12)),
                ),
                align="center",
            )

    ensure_min_margin(fig, min_top=int(max(80, 60 + max_y * 20)))


def add_hline_label(fig, y: float, text: str, line_color: str = "#999999", line_dash: str = "dash"):
    """Linea horizontal con etiqueta ubicada sin montar el encabezado."""
    fig.add_hline(y=float(y), line_dash=line_dash, line_color=line_color)
    fig.add_annotation(
        x=1,
        y=float(y),
        xref="paper",
        yref="y",
        text=text,
        showarrow=False,
        xanchor="right",
        xshift=-6,
        yshift=12,
        font=dict(color=line_color, size=11),
        align="right",
    )
