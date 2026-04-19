# =============================================================================
#  CASEN 2024 · Biobío & Ñuble · La brecha del campo vs la ciudad
#  Análisis de dependencia de subsidios estatales: Área Urbana vs Rural
#  Librería: Plotly Express + Graph Objects · Autor: ETL automatizado
# =============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ─────────────────────────────────────────────────────────────────────────────
# 0.  TEMPLATE GLOBAL  — Fuente corporativa, paleta y espaciado consistentes
# ─────────────────────────────────────────────────────────────────────────────
FONT_FAMILY = "Inter, Roboto, sans-serif"
COLOR_URBANO = "#1565C0"   # Azul oscuro  → ciudad
COLOR_RURAL  = "#2E7D32"   # Verde oscuro → campo
COLOR_SUBS   = "#C62828"   # Rojo         → subsidio estatal
COLOR_AUT    = "#1565C0"   # Azul         → ingreso autónomo
COLOR_BG     = "#FAFAFA"
COLOR_GRID   = "#EEEEEE"

PALETTE_REGION = {"Biobío": "#1565C0", "Ñuble": "#e65100"}

custom_tmpl = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY, color="#212121", size=12),
        paper_bgcolor="white",
        plot_bgcolor=COLOR_BG,
        title=dict(
            font=dict(family=FONT_FAMILY, size=18, color="#212121"),
            pad=dict(t=12, l=6),
            x=0.01,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E0E0E0",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=80, r=50, t=110, b=80),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="#BDBDBD",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID,
            gridwidth=1,
            zeroline=False,
            linecolor="#BDBDBD",
            tickfont=dict(size=11),
        ),
    )
)
pio.templates["casen_custom"] = custom_tmpl
pio.templates.default = "plotly_white+casen_custom"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DICCIONARIOS DE REFERENCIA  (extraídos del Libro de Códigos CASEN 2024)
# ─────────────────────────────────────────────────────────────────────────────
# Código CUT → nombre oficial de la comuna
COMUNAS = {
    8101:"Concepción",      8102:"Coronel",          8103:"Chiguayante",
    8104:"Florida",         8105:"Hualqui",           8106:"Lota",
    8107:"Penco",           8108:"San Pedro de la Paz",8109:"Santa Juana",
    8110:"Talcahuano",      8111:"Tomé",              8112:"Hualpén",
    8201:"Lebu",            8202:"Arauco",            8203:"Cañete",
    8204:"Contulmo",        8205:"Curanilahue",       8206:"Los Álamos",
    8207:"Tirúa",           8301:"Los Ángeles",       8302:"Antuco",
    8303:"Cabrero",         8304:"Laja",              8305:"Mulchén",
    8306:"Nacimiento",      8307:"Negrete",           8308:"Quilaco",
    8309:"Quilleco",        8310:"San Rosendo",       8311:"Santa Bárbara",
    8312:"Tucapel",         8313:"Yumbel",            8314:"Alto Biobío",
    16101:"Chillán",        16102:"Bulnes",            16103:"Chillán Viejo",
    16104:"El Carmen",      16105:"Pemuco",            16106:"Pinto",
    16107:"Quillón",        16108:"San Ignacio",       16109:"Yungay",
    16201:"Quirihue",       16202:"Cobquecura",        16203:"Coelemu",
    16204:"Ninhue",         16205:"Portezuelo",        16206:"Ranquil",
    16207:"Treguaco",       16301:"San Carlos",        16302:"Coihueco",
    16303:"Ñiquén",         16304:"San Fabián",        16305:"San Nicolás",
}

REGIONES = {8: "Biobío", 16: "Ñuble"}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ETL — Carga, cruce y enriquecimiento de variables
# ─────────────────────────────────────────────────────────────────────────────
print("Paso 1/3 · Cargando bases CASEN 2024...")
df_main = pd.read_stata("casen_2024.dta",              convert_categoricals=False)
df_geo  = pd.read_stata("casen_2024_provincia_comuna.dta", convert_categoricals=False)

# Cruce obligatorio por las dos llaves (folio + id_persona)
df = pd.merge(df_main, df_geo, on=["folio", "id_persona"], how="left")

# Filtro geográfico: solo Biobío (8) y Ñuble (16)
df = df[df["region"].isin([8, 16])].copy()

print("Paso 2/3 · Limpieza y cálculo de variables derivadas...")
cols_num = ["ytotcorh", "ysub", "yautcorh", "numper", "edad",
            "area", "pobreza", "region", "comuna"]
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Etiquetas descriptivas (libro de códigos CASEN)
df["nombre_region"] = df["region"].map(REGIONES)
df["nombre_comuna"] = df["comuna"].map(COMUNAS)

# area:  1 = Urbano  |  2 = Rural   (Fuente: HdR CASEN 2024 / Definición CENSO)
df["tipo_area"] = df["area"].map({1: "Urbano", 2: "Rural"})

# Métricas de ingreso
df["ingreso_pc"] = df["ytotcorh"] / df["numper"]
df["pct_subs"]   = np.where(
    df["ytotcorh"] > 0,
    (df["ysub"].fillna(0) / df["ytotcorh"]) * 100,
    np.nan
)
df["cond_pobreza"] = df["pobreza"].apply(
    lambda x: "En pobreza" if x in [1, 2] else "No pobre"
)

# ── Nivel hogar (1 fila por folio) ──
# Eliminar duplicados por folio para no sobreponderar hogares grandes
df_hog = df.drop_duplicates(subset=["folio"]).dropna(subset=["ingreso_pc"]).copy()

# Deciles de ingreso per cápita dentro de la zona para comparación justa
df_hog["decil"] = df_hog.groupby("tipo_area", observed=True)["ingreso_pc"].transform(
    lambda s: pd.qcut(s, 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
)

# Zona dominante por comuna (moda de area para cada código CUT)
zona_moda = (
    df_hog.groupby("nombre_comuna", observed=True)["area"]
    .agg(lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan)
    .reset_index()
)
zona_moda["zona_comunal"] = zona_moda["area"].map({1: "Urbano", 2: "Rural"})

print(f"Paso 3/3 · Hogares válidos: {len(df_hog):,} "
      f"(Biobío: {(df_hog['region']==8).sum():,} | Ñuble: {(df_hog['region']==16).sum():,})")
print("ETL completado.\n")

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 1 — Violín: Distribución de dependencia estatal por Área y Región
# Historia: "¿Tiene el campo más dependencia que la ciudad?"
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Gráfico 1: Violín multivariado Urbano/Rural × Región...")

# Solo hogares con algún subsidio recibido para graficar distribuciones reales
df_v = df_hog[df_hog["pct_subs"] > 0].dropna(subset=["tipo_area", "pct_subs"]).copy()

AREAS  = ["Urbano", "Rural"]
fig1 = go.Figure()

for area in AREAS:
    for region in ["Biobío", "Ñuble"]:
        datos = df_v[(df_v["tipo_area"] == region) & (df_v["nombre_region"] == region)]["pct_subs"]
        # reasignamos correctamente
        datos = df_v[(df_v["tipo_area"] == area) & (df_v["nombre_region"] == region)]["pct_subs"]

        if len(datos) < 10:
            continue

        fig1.add_trace(go.Violin(
            x=[area] * len(datos),
            y=datos,
            name=region,
            legendgroup=region,
            showlegend=(area == "Urbano"),          # leyenda solo una vez
            side="negative" if region == "Biobío" else "positive",
            line_color=PALETTE_REGION[region],
            fillcolor=PALETTE_REGION[region],
            opacity=0.65,
            meanline=dict(visible=True, color="white", width=2.5),
            box=dict(
                visible=True,
                line=dict(color="white", width=1.5),
                fillcolor="rgba(255,255,255,0.3)",
            ),
            points=False,          # suprimir puntos individuales: mejora Data-Ink Ratio
            spanmode="hard",
            hovertemplate=(
                f"<b>{region}</b> · %{{x}}<br>"
                "Dependencia subsidios: <b>%{y:.1f}%</b><br>"
                "<extra></extra>"
            ),
        ))

# Líneas de referencia: mediana por área
for area in AREAS:
    med = df_v[df_v["tipo_area"] == area]["pct_subs"].median()
    fig1.add_shape(
        type="line",
        x0=AREAS.index(area) - 0.45, x1=AREAS.index(area) + 0.45,
        y0=med, y1=med,
        line=dict(color="#757575", dash="dot", width=1.5),
    )
    fig1.add_annotation(
        x=area, y=med + 2.5,
        text=f"Mediana zona: <b>{med:.1f}%</b>",
        showarrow=False,
        font=dict(size=10, color="#424242"),
        bgcolor="rgba(255,255,255,0.8)",
    )

fig1.update_layout(
    title=dict(
        text=(
            "<b>¿Depende más el campo que la ciudad? Distribución de la dependencia estatal</b><br>"
            "<sup>% del ingreso del hogar proveniente de subsidios · Solo hogares que reciben algún subsidio<br>"
            "Biobío y Ñuble · CASEN 2024 · Vista izquierda = Biobío | Vista derecha = Ñuble</sup>"
        )
    ),
    violingap=0.06,
    violingroupgap=0,
    violinmode="overlay",
    xaxis=dict(title="Tipo de área (según variable <i>area</i> del CASEN/CENSO)", tickfont=dict(size=13)),
    yaxis=dict(
        title="% del ingreso total del hogar proveniente de subsidios estatales",
        ticksuffix="%",
        range=[0, 105],
    ),
    legend=dict(
        title="Región",
        orientation="h",
        yanchor="bottom", y=-0.22,
        xanchor="center", x=0.5,
    ),
    height=600,
)

fig1.write_image("gr1_violin_urbano_rural.png", width=1100, height=620, scale=2.5)
fig1.write_html("gr1_violin_urbano_rural.html", include_plotlyjs="cdn")
print("  -> gr1_violin_urbano_rural.png  (+ .html interactivo)\n")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 2 — Stacked Bar 100 %: Autonomía vs. Subsidio por Decil y Área
# Historia: "En los deciles más pobres del campo, el Estado es el principal proveedor"
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Gráfico 2: Barras apiladas 100% por decil y área...")

agg2 = (
    df_hog.dropna(subset=["decil", "tipo_area"])
    .groupby(["tipo_area", "decil"], observed=True)[["yautcorh", "ysub"]]
    .mean()
    .reset_index()
)
agg2["total"]    = (agg2["yautcorh"].fillna(0) + agg2["ysub"].fillna(0)).replace(0, np.nan)
agg2["pct_aut"]  = (agg2["yautcorh"].fillna(0) / agg2["total"] * 100).round(1)
agg2["pct_subs"] = (agg2["ysub"].fillna(0)     / agg2["total"] * 100).round(1)

fig2 = make_subplots(
    rows=1, cols=2,
    shared_yaxes=True,
    subplot_titles=["Zona Urbana", "Zona Rural"],
    horizontal_spacing=0.06,
)

DECIL_ORDER = [f"D{i}" for i in range(1, 11)]

for col_idx, area in enumerate(["Urbano", "Rural"], start=1):
    d = agg2[agg2["tipo_area"] == area].copy()
    d["decil"] = pd.Categorical(d["decil"], categories=DECIL_ORDER, ordered=True)
    d = d.sort_values("decil")

    # Barra: Ingreso Autónomo
    fig2.add_trace(go.Bar(
        x=d["decil"], y=d["pct_aut"],
        name="Ingreso autónomo",
        marker_color=COLOR_AUT,
        legendgroup="autonomo",
        showlegend=(col_idx == 1),
        hovertemplate="Decil: %{x}<br>Ingreso autónomo: <b>%{y:.1f}%</b><extra></extra>",
    ), row=1, col=col_idx)

    # Barra: Subsidios
    fig2.add_trace(go.Bar(
        x=d["decil"], y=d["pct_subs"],
        name="Subsidios estatales",
        marker_color=COLOR_SUBS,
        legendgroup="subsidios",
        showlegend=(col_idx == 1),
        hovertemplate="Decil: %{x}<br>Subsidios: <b>%{y:.1f}%</b><extra></extra>",
    ), row=1, col=col_idx)

    # Anotación obligatoria: % subsidio en D1 (los más vulnerables)
    d1 = d[d["decil"] == "D1"]
    if not d1.empty:
        pct_d1 = d1["pct_subs"].values[0]
        xref = "x"       if col_idx == 1 else "x2"
        yref = "y"
        fig2.add_annotation(
            x="D1", y=d1["pct_aut"].values[0] + pct_d1 / 2,
            text=f"<b>{pct_d1:.0f}%<br>subsidio</b>",
            font=dict(size=12, color="white", family=FONT_FAMILY),
            showarrow=False,
            xref=xref, yref=yref,
        )

fig2.update_layout(
    barmode="stack",
    title=dict(
        text=(
            "<b>Composición del ingreso: ¿quién sostiene a los hogares más pobres?</b><br>"
            "<sup>Proporción de ingreso autónomo vs. subsidios estatales por decil de ingreso · "
            "Panel Izquierdo = Urbano | Panel Derecho = Rural · CASEN 2024</sup>"
        )
    ),
    yaxis=dict(title="Composición del ingreso (%)", ticksuffix="%", range=[0, 108]),
    yaxis2=dict(ticksuffix="%", range=[0, 108]),
    xaxis=dict(title="Decil de ingreso per cápita (D1 = más pobre · D10 = más rico)"),
    xaxis2=dict(title="Decil de ingreso per cápita (D1 = más pobre · D10 = más rico)"),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.22,
        xanchor="center", x=0.5,
        font=dict(size=12),
    ),
    height=540,
)

fig2.write_image("gr2_stacked_decil_area.png", width=1300, height=560, scale=2.5)
fig2.write_html("gr2_stacked_decil_area.html", include_plotlyjs="cdn")
print("  -> gr2_stacked_decil_area.png  (+ .html interactivo)\n")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 3 — Mapa de Calor Matricial: Dependencia por Comuna y Ruralidad
# Historia: "La brecha del campo: ¿Qué comunas dependen más del Estado?"
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Gráfico 3: Mapa de calor matricial comunal...")

df_com = (
    df_hog.dropna(subset=["nombre_comuna", "pct_subs"])
    .groupby("nombre_comuna", observed=True)
    .agg(
        pct_dep   =("pct_subs",    "mean"),
        ing_medio =("ytotcorh",    "mean"),
        n_hogares =("folio",       "count"),
        zona_area =("area",        lambda s: s.mode().iloc[0]),   # zona dominante
    )
    .reset_index()
    .sort_values("pct_dep")      # orden ascendente para eje Y
)
df_com["pct_dep_r"]   = df_com["pct_dep"].round(1)
df_com["zona_label"]  = df_com["zona_area"].map({1: "Urbano", 2: "Rural"})
df_com["nombre_zona"] = df_com.apply(
    lambda r: f"  {r['nombre_comuna']} <b>[Rural]</b>" if r["zona_label"] == "Rural"
              else f"  {r['nombre_comuna']}",
    axis=1,
)

# Color por ingreso medio normalizado
ing_min, ing_max = df_com["ing_medio"].min(), df_com["ing_medio"].max()
df_com["ing_norm"] = (df_com["ing_medio"] - ing_min) / (ing_max - ing_min + 1)

fig3 = go.Figure()

# Barras principales coloreadas por % dependencia (YlOrRd)
fig3.add_trace(go.Bar(
    x=df_com["pct_dep_r"],
    y=df_com["nombre_zona"],
    orientation="h",
    marker=dict(
        color=df_com["pct_dep_r"],
        colorscale="YlOrRd",
        cmin=df_com["pct_dep_r"].min(),
        cmax=df_com["pct_dep_r"].max(),
        showscale=True,
        colorbar=dict(
            title=dict(text="% dependencia<br>promedio", side="right", font=dict(size=11)),
            ticksuffix="%",
            thickness=14,
            len=0.55,
            yanchor="middle", y=0.5,
            outlinewidth=0,
        ),
        line=dict(width=0),
    ),
    text=df_com["pct_dep_r"].apply(lambda v: f"{v:.1f}%"),
    textposition="outside",
    textfont=dict(size=10, family=FONT_FAMILY),
    customdata=df_com[["zona_label","ing_medio","n_hogares","nombre_comuna"]].values,
    hovertemplate=(
        "<b>%{customdata[3]}</b> (%{customdata[0]})<br>"
        "Dependencia promedio: <b>%{x:.1f}%</b><br>"
        "Ingreso medio del hogar: <b>$%{customdata[1]:,.0f}</b><br>"
        "Hogares en la muestra: %{customdata[2]}<br>"
        "<extra></extra>"
    ),
))

# Marcadores adicionales sobre comunas rurales (diamante rojo)
df_rural = df_com[df_com["zona_label"] == "Rural"]
fig3.add_trace(go.Scatter(
    x=df_rural["pct_dep_r"] + 0.6,
    y=df_rural["nombre_zona"],
    mode="markers",
    marker=dict(symbol="diamond", size=9, color="#b71c1c", opacity=0.85),
    name="Zona Rural (dominante)",
    hoverinfo="skip",
))

# Línea de promedio total
promedio_total = df_com["pct_dep_r"].mean()
fig3.add_vline(
    x=promedio_total,
    line_dash="dash",
    line_color="#616161",
    line_width=1.5,
    annotation_text=f"<b>Promedio zona: {promedio_total:.1f}%</b>",
    annotation_position="top right",
    annotation_font=dict(size=10, color="#424242"),
    annotation_bgcolor="rgba(255,255,255,0.8)",
)

fig3.update_layout(
    title=dict(
        text=(
            "<b>La brecha del campo: ¿Qué comunas dependen más del Estado?</b><br>"
            "<sup>Dependencia promedio de subsidios estatales sobre el ingreso total del hogar · "
            "Ordenado de menor a mayor dependencia<br>"
            "Biobío y Ñuble · CASEN 2024 · "
            "<span style='color:#b71c1c'>♦ = Zona Rural dominante</span>    "
            "Texto en negrita [Rural] = confirmado por variable <i>area</i> del CASEN</sup>"
        )
    ),
    xaxis=dict(
        title="% promedio del ingreso del hogar que proviene de subsidios del Estado",
        ticksuffix="%",
        range=[0, df_com["pct_dep_r"].max() * 1.22],
        showgrid=False,
    ),
    yaxis=dict(
        title="",
        autorange=True,
        tickfont=dict(size=10.5),
        showgrid=False,
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.07,
        xanchor="right", x=1,
        font=dict(size=11),
    ),
    height=1150,
    margin=dict(l=210, r=120, t=130, b=60),
)

fig3.write_image("gr3_comunas_dependencia.png", width=1050, height=1150, scale=2.5)
fig3.write_html("gr3_comunas_dependencia.html", include_plotlyjs="cdn")
print("  -> gr3_comunas_dependencia.png  (+ .html interactivo)\n")

print("=" * 60)
print("Todos los graficos generados y exportados correctamente.")
print("  PNG (300 dpi equiv.): gr1, gr2, gr3")
print("  HTML interactivos:    gr1, gr2, gr3")
print("=" * 60)
