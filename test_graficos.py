# =============================================================================
# ENTREGABLE 1 – Dependencia Estatal en Biobío y Ñuble | CASEN 2024
# Visualizaciones con Plotly Express + Graph Objects
# =============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ─────────────────────────────────────────────────────────────────────────────
# 0.  TEMPLATE GLOBAL  (evita styling ad-hoc en cada figura)
# ─────────────────────────────────────────────────────────────────────────────
FUENTE       = "Inter, sans-serif"
COLOR_AZUL   = "#1a6faf"   # Ingreso autónomo
COLOR_ROJO   = "#d63031"   # Subsidios / alta dependencia
COLOR_VERDE  = "#00897b"   # No pobre
COLOR_GRIS   = "#636e72"
TITULO_SIZE  = 17
TICK_SIZE    = 11

template_custom = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FUENTE, color="#2d3436"),
        paper_bgcolor="white",
        plot_bgcolor="#f9f9f9",
        colorway=[COLOR_AZUL, COLOR_ROJO, "#fdcb6e", COLOR_VERDE, "#a29bfe"],
        title=dict(font=dict(size=TITULO_SIZE, family=FUENTE), pad=dict(t=10, l=4)),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#dfe6e9",
                    borderwidth=1, font=dict(size=12)),
        margin=dict(l=70, r=40, t=90, b=70),
        xaxis=dict(showgrid=False, linecolor="#b2bec3", tickfont=dict(size=TICK_SIZE)),
        yaxis=dict(gridcolor="#ecf0f1", linecolor="#b2bec3", tickfont=dict(size=TICK_SIZE)),
    )
)
pio.templates["custom"] = template_custom
pio.templates.default  = "plotly_white+custom"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DICCIONARIOS DE REFERENCIA
# ─────────────────────────────────────────────────────────────────────────────
COMUNAS = {
    8101:"Concepción",8102:"Coronel",8103:"Chiguayante",8104:"Florida",
    8105:"Hualqui",8106:"Lota",8107:"Penco",8108:"San Pedro de la Paz",
    8109:"Santa Juana",8110:"Talcahuano",8111:"Tomé",8112:"Hualpén",
    8201:"Lebu",8202:"Arauco",8203:"Cañete",8204:"Contulmo",
    8205:"Curanilahue",8206:"Los Álamos",8207:"Tirúa",
    8301:"Los Ángeles",8302:"Antuco",8303:"Cabrero",8304:"Laja",
    8305:"Mulchén",8306:"Nacimiento",8307:"Negrete",8308:"Quilaco",
    8309:"Quilleco",8310:"San Rosendo",8311:"Santa Bárbara",
    8312:"Tucapel",8313:"Yumbel",8314:"Alto Biobío",
    16101:"Chillán",16102:"Bulnes",16103:"Chillán Viejo",16104:"El Carmen",
    16105:"Pemuco",16106:"Pinto",16107:"Quillón",16108:"San Ignacio",
    16109:"Yungay",16201:"Quirihue",16202:"Cobquecura",16203:"Coelemu",
    16204:"Ninhue",16205:"Portezuelo",16206:"Ranquil",16207:"Treguaco",
    16301:"San Carlos",16302:"Coihueco",16303:"Ñiquén",
    16304:"San Fabián",16305:"San Nicolás",
}

# Comunas con predominancia rural (sin capital provincial o densidad < 50 hab/km²)
RURALES = {
    8104,8105,8109,8204,8207,8302,8304,8307,8308,8309,
    8310,8311,8312,8313,8314,16103,16104,16105,16106,
    16107,16108,16109,16202,16203,16204,16205,16206,
    16207,16302,16303,16304,16305,
}

REGIONES = {8:"Biobío", 16:"Ñuble"}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CARGA Y ETL
# ─────────────────────────────────────────────────────────────────────────────
print("Cargando datos CASEN 2024 (puede tardar ~30 s)...")
df_p = pd.read_stata("casen_2024.dta", convert_categoricals=False)
df_c = pd.read_stata("casen_2024_provincia_comuna.dta", convert_categoricals=False)
df   = pd.merge(df_p, df_c, on=["folio","id_persona"], how="left")
df   = df[df["region"].isin([8,16])].copy()

for col in ["ytotcorh","ysub","yautcorh","numper","edad","area","pobreza",
            "region","comuna","expr"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Variables derivadas
df["nombre_region"]  = df["region"].map(REGIONES)
df["nombre_comuna"]  = df["comuna"].map(COMUNAS)
df["tipo_zona"]      = df["area"].map({1:"Urbano", 2:"Rural"})
df["zona_comunal"]   = df["comuna"].isin(RURALES).map({True:"Rural", False:"Urbano"})
df["ingreso_pc"]     = df["ytotcorh"] / df["numper"]
df["porc_subs"]      = np.where(
    df["ytotcorh"] > 0,
    (df["ysub"].fillna(0) / df["ytotcorh"]) * 100,
    np.nan
)
df["cond_pobreza"]   = df["pobreza"].apply(
    lambda x: "En pobreza" if x in [1,2] else "No pobre"
)

# Nivel hogar (un registro por folio)
df_hog = (
    df.drop_duplicates(subset=["folio"])
      .dropna(subset=["ingreso_pc"])
      .copy()
)
# Deciles calculados sobre todos los hogares de la zona
df_hog["decil"] = pd.qcut(
    df_hog["ingreso_pc"], 10,
    labels=[f"D{i}" for i in range(1,11)],
    duplicates="drop"
)

print(f"Hogares en Biobio y Nuble: {len(df_hog):,}")
print("ETL completado.\n")

# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 1 – Stacked Bar 100 %: Composicion del ingreso por decil y region
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Grafico 1...")

agg1 = (
    df_hog.groupby(["nombre_region","decil"], observed=True)[["yautcorh","ysub"]]
    .mean().reset_index()
)
agg1["total"]    = agg1[["yautcorh","ysub"]].sum(axis=1).replace(0, np.nan)
agg1["pct_aut"]  = (agg1["yautcorh"] / agg1["total"] * 100).round(1)
agg1["pct_sub"]  = (agg1["ysub"]     / agg1["total"] * 100).round(1)

# Long-format para px.bar
long1 = pd.melt(
    agg1, id_vars=["nombre_region","decil","total"],
    value_vars=["pct_aut","pct_sub"],
    var_name="tipo_ingreso", value_name="porcentaje"
)
long1["tipo_ingreso"] = long1["tipo_ingreso"].map({
    "pct_aut": "Ingreso autónomo (trabajo, pensión propia)",
    "pct_sub": "Subsidios y transferencias del Estado"
})

fig1 = px.bar(
    long1,
    x="decil", y="porcentaje",
    color="tipo_ingreso",
    barmode="stack",
    facet_col="nombre_region",
    facet_col_spacing=0.08,
    color_discrete_map={
        "Ingreso autónomo (trabajo, pensión propia)": COLOR_AZUL,
        "Subsidios y transferencias del Estado":       COLOR_ROJO,
    },
    labels={
        "decil":"Decil de ingreso per cápita",
        "porcentaje":"Composición del ingreso (%)",
        "tipo_ingreso":"Fuente de ingreso",
    },
    custom_data=["porcentaje","tipo_ingreso","nombre_region"],
)

# Tooltip enriquecido
fig1.update_traces(
    hovertemplate=(
        "<b>%{customdata[2]}</b><br>"
        "Decil: %{x}<br>"
        "%{customdata[1]}: <b>%{customdata[0]:.1f}%</b><extra></extra>"
    )
)

# Anotación directa sobre el primer y último decil (subsidio)
subs_rows = agg1[agg1["decil"].isin(["D1","D10"])].copy()
for _, row in subs_rows.iterrows():
    col_idx = 1 if row["nombre_region"] == "Biobío" else 2
    xref = "x" if col_idx == 1 else "x2"
    yref = "y"
    fig1.add_annotation(
        x=row["decil"], y=row["pct_aut"] + row["pct_sub"]/2,
        text=f"<b>{row['pct_sub']:.0f}%\nsubsidio</b>",
        font=dict(size=10, color="white"),
        showarrow=False,
        xref=xref, yref=yref,
    )

fig1.update_layout(
    title=dict(
        text=(
            "<b>¿De qué viven los hogares? Ingreso propio vs. subsidios del Estado</b><br>"
            "<sup>Biobío y Ñuble · CASEN 2024 · "
            "D1 = hogares más pobres · D10 = hogares más ricos</sup>"
        )
    ),
    yaxis=dict(range=[0,105], ticksuffix="%"),
    yaxis2=dict(range=[0,105], ticksuffix="%"),
    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    height=520,
)
# Eliminar prefijos automáticos de facets
fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig1.write_image("grafico_1_composicion_decil.png", width=1200, height=540, scale=2.5)
print("  -> grafico_1_composicion_decil.png guardado")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 2 – Barras horizontales por comuna con escala de color secuencial
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Grafico 2...")

df2 = (
    df_hog.groupby(["nombre_region","nombre_comuna","zona_comunal"], observed=True)
    ["porc_subs"].mean().reset_index()
    .dropna()
    .sort_values("porc_subs")
)
df2["pct_round"] = df2["porc_subs"].round(1)
df2["etiqueta"] = df2.apply(
    lambda r: f"{r['nombre_comuna']}  ({'Rural' if r['zona_comunal']=='Rural' else 'Urbano'})",
    axis=1
)
df2["marker_sym"] = df2["zona_comunal"].map({"Rural":"diamond","Urbano":"circle"})

avg_zona = df2["porc_subs"].mean()

fig2 = go.Figure()

for zona, group in df2.groupby("zona_comunal", observed=True):
    color_bar = "#e17055" if zona == "Rural" else "#74b9ff"
    fig2.add_trace(go.Bar(
        x=group["pct_round"],
        y=group["etiqueta"],
        orientation="h",
        name=f"Zona {zona}",
        marker=dict(
            color=group["pct_round"],
            colorscale="YlOrRd",
            cmin=df2["pct_round"].min(),
            cmax=df2["pct_round"].max(),
            showscale=False,
            line=dict(width=0),
        ),
        text=group["pct_round"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(size=9.5),
        customdata=group[["nombre_region","zona_comunal","pct_round"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Region: %{customdata[0]}<br>"
            "Zona: %{customdata[1]}<br>"
            "Dependencia promedio: <b>%{customdata[2]:.1f}%</b>"
            "<extra></extra>"
        ),
    ))

# Linea de promedio
fig2.add_vline(
    x=avg_zona, line_dash="dash", line_color="#636e72", line_width=1.5,
    annotation_text=f"Promedio zona<br><b>{avg_zona:.1f}%</b>",
    annotation_position="top right",
    annotation_font=dict(size=10, color="#636e72"),
)

# Colorbar independiente
fig2.add_trace(go.Bar(
    x=[None], y=[None], name="",
    marker=dict(
        color=[0], colorscale="YlOrRd",
        cmin=df2["pct_round"].min(), cmax=df2["pct_round"].max(),
        colorbar=dict(
            title=dict(text="% dependencia<br>promedio", side="right"),
            thickness=14, len=0.5, yanchor="middle", y=0.5,
        ),
        showscale=True,
    ),
    showlegend=False,
))

fig2.update_layout(
    title=dict(
        text=(
            "<b>¿Qué comunas dependen más del Estado?</b><br>"
            "<sup>Porcentaje promedio del ingreso del hogar proveniente de subsidios "
            "· Biobío y Ñuble · CASEN 2024<br>"
            "Comunas ordenadas de menor a mayor dependencia "
            "· Zonas: Urbana vs Rural</sup>"
        )
    ),
    xaxis=dict(title="% del ingreso que proviene de subsidios", ticksuffix="%", range=[0, df2["pct_round"].max()*1.18]),
    yaxis=dict(title="", autorange=True, tickfont=dict(size=10.5)),
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
    height=1100,
    margin=dict(l=230, r=100, t=110, b=60),
)

fig2.write_image("grafico_2_dependencia_comuna.png", width=1000, height=1100, scale=2.5)
print("  -> grafico_2_dependencia_comuna.png guardado")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 3 – Scatter multivariado: tamaño hogar × subsidios × pobreza × region
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Grafico 3...")

df3 = df_hog.dropna(subset=["ysub","numper","cond_pobreza","porc_subs"]).copy()
df3 = df3[df3["ysub"] > 0]
df3 = df3.sample(n=min(5000, len(df3)), random_state=42)

# Agregar jitter leve en X para evitar superposicion
rng = np.random.default_rng(42)
df3["numper_j"] = df3["numper"] + rng.uniform(-0.2, 0.2, len(df3))

fig3 = px.scatter(
    df3,
    x="numper_j",
    y="ysub",
    color="porc_subs",
    symbol="nombre_region",
    facet_col="cond_pobreza",
    facet_col_spacing=0.08,
    log_y=True,
    opacity=0.65,
    color_continuous_scale="RdYlGn_r",   # escala divergente: verde=baja dep, rojo=alta
    range_color=[0, 100],
    size_max=9,
    labels={
        "numper_j"  : "Numero de personas en el hogar",
        "ysub"      : "Monto mensual de subsidios recibidos ($ CLP, escala log.)",
        "porc_subs" : "% del ingreso = subsidios",
        "nombre_region": "Region",
        "cond_pobreza":  "Condicion de pobreza",
    },
    custom_data=["nombre_comuna","nombre_region","numper","ysub","porc_subs","cond_pobreza"],
)

fig3.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
        "Personas en el hogar: %{customdata[2]}<br>"
        "Subsidios mensuales: <b>$%{customdata[3]:,.0f}</b><br>"
        "Dependencia del Estado: <b>%{customdata[4]:.1f}%</b><br>"
        "Condicion: %{customdata[5]}<extra></extra>"
    ),
    marker=dict(size=6, line=dict(width=0)),
)
fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig3.update_layout(
    title=dict(
        text=(
            "<b>Subsidios, tamaño del hogar y pobreza: ¿quienes reciben más apoyo estatal?</b><br>"
            "<sup>Hogares de Biobío y Ñuble que reciben algún subsidio · CASEN 2024 · "
            "Color = % del ingreso que representa el subsidio (verde=bajo, rojo=alto)</sup>"
        )
    ),
    coloraxis_colorbar=dict(
        title="% del ingreso<br>= subsidios",
        ticksuffix="%",
        thickness=14,
    ),
    height=580,
    legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
)

fig3.write_image("grafico_3_subsidios_pobreza.png", width=1300, height=600, scale=2.5)
print("  -> grafico_3_subsidios_pobreza.png guardado")


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICO 4 – Violin: dependencia por rango etario del jefe de hogar
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Grafico 4...")

jefe_col = "pco1" if "pco1" in df.columns else None
df4 = df[df[jefe_col] == 1].copy() if jefe_col else df.drop_duplicates("folio").copy()

bins_edad   = [17, 29, 44, 59, 100]
labels_edad = ["Jovenes (18-29)", "Adultos (30-44)", "Adultos Mayores (45-59)", "Tercera Edad (60+)"]
df4["rango_etario"] = pd.cut(df4["edad"], bins=bins_edad, labels=labels_edad, right=True)
df4 = df4.dropna(subset=["rango_etario","porc_subs"])
df4 = df4[df4["porc_subs"] > 0]

COLORES_REGION = {"Biobío": COLOR_AZUL, "Ñuble": COLOR_ROJO}

fig4 = go.Figure()

for region in ["Biobío","Ñuble"]:
    sub = df4[df4["nombre_region"] == region]
    for cat in labels_edad:
        datos = sub[sub["rango_etario"] == cat]["porc_subs"]
        if len(datos) < 10:
            continue
        fig4.add_trace(go.Violin(
            x=[cat] * len(datos),
            y=datos,
            name=region,
            legendgroup=region,
            showlegend=(cat == labels_edad[0]),
            side="negative" if region == "Biobío" else "positive",
            line_color=COLORES_REGION[region],
            fillcolor=COLORES_REGION[region],
            opacity=0.6,
            meanline_visible=True,
            box_visible=True,
            points=False,
            spanmode="hard",
            hovertemplate=(
                f"<b>{region}</b><br>"
                "Rango: %{x}<br>"
                "% dependencia: %{y:.1f}%<extra></extra>"
            ),
        ))

# Anotacion PGU en Tercera Edad
fig4.add_annotation(
    x="Tercera Edad (60+)", y=96,
    text=(
        "<b>La PGU</b> (Pension Garantizada Universal)<br>"
        "es el principal subsidio en este grupo.<br>"
        "Representa >80% del ingreso en muchos hogares."
    ),
    showarrow=True, arrowhead=2, arrowcolor=COLOR_GRIS,
    ax=110, ay=-70,
    font=dict(size=10, color="#2d3436"),
    bgcolor="#ffeaa7", bordercolor="#fdcb6e", borderwidth=1,
    borderpad=6,
)

fig4.update_layout(
    title=dict(
        text=(
            "<b>¿Depende mas del Estado quien esta mas cerca de la vejez?</b><br>"
            "<sup>Distribucion del % del ingreso que proviene de subsidios, "
            "segun edad del jefe/a de hogar · Biobio y Nuble · CASEN 2024<br>"
            "Lineas internas: mediana y cuartiles (25 %, 75 %) — "
            "Vista izquierda = Biobio | Vista derecha = Nuble</sup>"
        )
    ),
    violingap=0.05,
    violingroupgap=0,
    violinmode="overlay",
    xaxis=dict(title="Rango etario del jefe/a de hogar"),
    yaxis=dict(title="% del ingreso del hogar proveniente de subsidios", ticksuffix="%"),
    legend=dict(
        title="Region", orientation="h",
        yanchor="bottom", y=-0.18, xanchor="center", x=0.5
    ),
    height=600,
)

fig4.write_image("grafico_4_violin_edad.png", width=1200, height=620, scale=2.5)
print("  -> grafico_4_violin_edad.png guardado\n")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTAR TAMBIEN COMO HTML (interactivo)
# ─────────────────────────────────────────────────────────────────────────────
for fig_obj, nombre in [
    (fig1, "grafico_1_composicion_decil"),
    (fig2, "grafico_2_dependencia_comuna"),
    (fig3, "grafico_3_subsidios_pobreza"),
    (fig4, "grafico_4_violin_edad"),
]:
    fig_obj.write_html(f"{nombre}.html", include_plotlyjs="cdn")
    print(f"  -> {nombre}.html guardado (version interactiva)")

print("\nTodos los graficos generados correctamente.")
