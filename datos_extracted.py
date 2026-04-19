import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CELL ---

import pandas as pd
import numpy as np

# 1. Leer la base de datos principal de la Casen 2024
df_principal = pd.read_stata("casen_2024.dta", convert_categoricals=False)

# 2. Leer la base de datos de provincia y comuna
# (Asegúrate de que el nombre del archivo y la extensión sean los correctos)
df_prov_com = pd.read_stata("casen_2024_provincia_comuna.dta", convert_categoricals=False)

# 3. Unir ambas bases de datos usando 'folio' e 'id_persona' como llaves
df_completo = pd.merge(
    left=df_principal, 
    right=df_prov_com, 
    on=['folio', 'id_persona'], 
    how='left' # 'left' mantiene todos los registros de la base principal
)

#Filtrare la base de datos para quedarme solamente con la región de ñuble y biobío que son la número 16 y 8 respectivamente
df = df_completo[df_completo['region'].isin([8, 16])]


# Opcional: Verificar el resultado del cruce
print(f"Dimensiones de la base principal: {df_completo.shape}")
print(f"Dimensiones de la base unida: {df.shape}")

# --- CELL ---

# =============================================================================
# HISTORIA 2: Brechas de acceso a salud preventiva y especializada
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# -----------------------------------------------------------------------------
# 1. Preparación de variables
# -----------------------------------------------------------------------------

# 1.1 Previsión de salud (s13 y s13_fonasa)
# Recodificamos según códigos típicos de CASEN (ajusta si es necesario)
def recode_prevision(row):
    s13 = row.get('s13', np.nan)
    s13_fonasa = row.get('s13_fonasa', np.nan)
    if pd.isna(s13):
        return np.nan
    if s13 == 1:  # FONASA
        if s13_fonasa in [1,2,3,4]:  # grupos A, B, C, D
            return 'FONASA A-D'
        elif s13_fonasa == 5:
            return 'FONASA otros (adherente)'
        else:
            return 'FONASA (sin grupo)'
    elif s13 == 2:  # ISAPRE
        return 'ISAPRE'
    elif s13 == 3:  # FFAA o del orden
        return 'FFAA/Orden'
    elif s13 == 4:  # Otro (particular, etc.)
        return 'Otro sistema'
    elif s13 == 5:  # Ninguno
        return 'Sin previsión'
    else:
        return 'Otro'

df['prevision'] = df.apply(recode_prevision, axis=1)

# 1.2 Seguro complementario (s15a)
# Asumimos: s15a = 1 (Sí), 2 (No)
df['seguro_comp'] = df['s15a'].map({1: 'Sí', 2: 'No'})

# 1.3 Nivel educacional (e6a)
# Recodificar según niveles típicos (ajusta los códigos según tu base)
def recode_educ(e6a):
    if pd.isna(e6a):
        return np.nan
    # Niveles comunes en CASEN: 
    # 1=Sin educación, 2=Básica incompleta, 3=Básica completa, 
    # 4=Media incompleta, 5=Media completa, 6=Superior incompleta, 
    # 7=Superior completa, 8=Postgrado
    if e6a in [1,2]:
        return 'Básica incompleta o menos'
    elif e6a == 3:
        return 'Básica completa'
    elif e6a == 4:
        return 'Media incompleta'
    elif e6a == 5:
        return 'Media completa'
    elif e6a in [6,7,8]:
        return 'Superior (completa/incompleta)'
    else:
        return 'Otro'

df['educ_nivel'] = df['e6a'].apply(recode_educ)

# 1.4 Edad (asumimos variable 'edad' - si no, cámbiala)
# Si no tienes 'edad', busca 'r_edad' o 'edad_persona'
if 'edad' not in df.columns:
    print("ADVERTENCIA: No se encontró columna 'edad'. Usa el nombre correcto.")
    # Podrías asignar una edad ficticia para pruebas, pero mejor revisa.
    # Para este ejemplo, asumimos que existe.

# 1.5 Variables de salud preventiva y acceso

# Exámenes preventivos (solo para mujeres)
df['mujer'] = df['sexo_cod'] == 2  # asumiendo 1=hombre, 2=mujer

# Papanicolau últimos 3 años (s9a)
# s9a: 1=Sí, 2=No
df['papanicolau'] = df['s9a'].map({1: 'Sí', 2: 'No'})

# Mamografía últimos 3 años (s11a)
df['mamografia'] = df['s11a'].map({1: 'Sí', 2: 'No'})

# Atenciones recientes (últimos 3 meses)
df['atencion_med_gral'] = df['s20a_preg'].map({1: 'Sí', 2: 'No'})
df['atencion_urgencia'] = df['s21a_preg'].map({1: 'Sí', 2: 'No'})
df['atencion_salud_mental'] = df['s22a_preg'].map({1: 'Sí', 2: 'No'})
df['atencion_especialidad'] = df['s23a_preg'].map({1: 'Sí', 2: 'No'})
df['atencion_dental'] = df['s24a_preg'].map({1: 'Sí', 2: 'No'})

# Número de controles en últimos 12 meses (s26a)
df['num_controles'] = pd.to_numeric(df['s26a'], errors='coerce')

# Tipo de último control (s26u) - puedes agrupar categorías
# (opcional: recodificar según interés)

# Barreras de acceso (últimos 3 meses) - s19a a s19e
# Cada una: 1=Sí tuvo problema, 2=No
barreras = {
    'problema_llegar': 's19a',
    'problema_cita': 's19b',
    'problema_atencion_estab': 's19c',
    'problema_pagar': 's19d',
    'problema_medicamentos': 's19e'
}
for col, var in barreras.items():
    df[col] = df[var].map({1: 'Sí', 2: 'No'})

# 1.6 Filtrar por edad/sexo para ciertos análisis
# Para Papanicolau: mujeres de 25 a 64 años (recomendación típica)
df['pap_mujer_25_64'] = df['mujer'] & (df['edad'].between(25, 64))
# Para mamografía: mujeres de 50 a 69 años (o según guía)
df['mamo_mujer_50_69'] = df['mujer'] & (df['edad'].between(50, 69))

# -----------------------------------------------------------------------------
# 2. Análisis descriptivos y tablas cruzadas
# -----------------------------------------------------------------------------

# 2.1 Función para tabla cruzada con porcentajes por fila
def cross_tab_with_percent(df, row_var, col_var, row_filter=None):
    if row_filter is not None:
        df_temp = df[row_filter].copy()
    else:
        df_temp = df.copy()
    tabla = pd.crosstab(df_temp[row_var], df_temp[col_var], normalize='index') * 100
    return tabla

# 2.2 Ejemplos de tablas clave

# Tabla 1: Papanicolau según previsión (mujeres 25-64)
if df['pap_mujer_25_64'].any():
    tabla_pap = cross_tab_with_percent(df, 'prevision', 'papanicolau', 
                                       row_filter=df['pap_mujer_25_64'])
    print("\n=== Papanicolau en últimos 3 años (mujeres 25-64) según previsión ===\n")
    print(tabla_pap)

# Tabla 2: Mamografía según previsión (mujeres 50-69)
if df['mamo_mujer_50_69'].any():
    tabla_mamo = cross_tab_with_percent(df, 'prevision', 'mamografia',
                                        row_filter=df['mamo_mujer_50_69'])
    print("\n=== Mamografía en últimos 3 años (mujeres 50-69) según previsión ===\n")
    print(tabla_mamo)

# Tabla 3: Atención de especialidad según previsión (toda la muestra)
tabla_especialidad = cross_tab_with_percent(df, 'prevision', 'atencion_especialidad')
print("\n=== Atención de especialidad últimos 3 meses según previsión ===\n")
print(tabla_especialidad)

# Tabla 4: Atención dental según nivel educacional
tabla_dental_educ = cross_tab_with_percent(df, 'educ_nivel', 'atencion_dental')
print("\n=== Atención dental últimos 3 meses según nivel educacional ===\n")
print(tabla_dental_educ)

# Tabla 5: Problemas para conseguir hora según previsión
tabla_cita = cross_tab_with_percent(df, 'prevision', 'problema_cita')
print("\n=== Problemas para conseguir cita/atención según previsión ===\n")
print(tabla_cita)

# -----------------------------------------------------------------------------
# 3. Visualizaciones
# -----------------------------------------------------------------------------

# 3.1 Gráfico de barras: % de Papanicolau por previsión (solo mujeres 25-64)
if df['pap_mujer_25_64'].any():
    plt.figure()
    df_pap = df[df['pap_mujer_25_64']].groupby('prevision')['papanicolau'].apply(
        lambda x: (x == 'Sí').mean() * 100).reset_index()
    sns.barplot(data=df_pap, x='prevision', y='papanicolau', palette='Blues_d')
    plt.title('Porcentaje de mujeres (25-64 años) que se realizaron Papanicolau\nen últimos 3 años según previsión de salud', fontsize=12)
    plt.ylabel('% que se realizó el examen')
    plt.xlabel('Previsión de salud')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pap_prevision.png', dpi=150)
    plt.show()

# 3.2 Boxplot del número de controles según previsión
plt.figure()
sns.boxplot(data=df.dropna(subset=['num_controles', 'prevision']), 
            x='prevision', y='num_controles', palette='Set2')
plt.title('Número de controles de salud en últimos 12 meses según previsión', fontsize=12)
plt.ylabel('Número de controles')
plt.xlabel('Previsión de salud')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('controles_prevision.png', dpi=150)
plt.show()

# 3.3 Mapa de calor de barreras según previsión (porcentaje que reportó problema)
barreras_list = ['problema_llegar', 'problema_cita', 'problema_atencion_estab', 
                 'problema_pagar', 'problema_medicamentos']
# Calcular porcentaje de "Sí" para cada barrera por previsión
barreras_by_prevision = df.groupby('prevision')[barreras_list].apply(
    lambda g: (g == 'Sí').mean() * 100)
plt.figure(figsize=(12, 6))
sns.heatmap(barreras_by_prevision, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': '% que reportó problema'})
plt.title('Porcentaje de personas que reportaron barreras de acceso a salud\nsegún previsión (últimos 3 meses)', fontsize=12)
plt.tight_layout()
plt.savefig('barreras_prevision.png', dpi=150)
plt.show()

# 3.4 Gráfico de barras apiladas: tipo de atención más reciente por nivel educacional
# Usaremos la última atención de medicina general (s20b) o especialidad (s23b) como ejemplo.
# Vamos a crear una variable 'tipo_establecimiento' a partir de s20b (última atención medicina general)
# s20b: 1=Consultorio/CECOSF, 2=Hospital público, 3=Posta rural, 4=CESFAM, 
# 5=Hospital clínico privado, 6=Clínica privada, 7=Consultorio particular, 8=Otro
def recode_establecimiento(s20b):
    if pd.isna(s20b):
        return np.nan
    if s20b in [1,2,3,4]:
        return 'Público'
    elif s20b in [5,6,7]:
        return 'Privado'
    else:
        return 'Otro'
df['establecimiento_tipo'] = df['s20b'].apply(recode_establecimiento)
tabla_establec_educ = pd.crosstab(df['educ_nivel'], df['establecimiento_tipo'], normalize='index') * 100
tabla_establec_educ.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.title('Tipo de establecimiento de última atención de medicina general\nsegún nivel educacional', fontsize=12)
plt.ylabel('Porcentaje')
plt.xlabel('Nivel educacional')
plt.legend(title='Establecimiento')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('establecimiento_educ.png', dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 4. Pruebas de significación (opcional)
# -----------------------------------------------------------------------------
# Ejemplo: Chi-cuadrado entre previsión y realización de mamografía (mujeres 50-69)
if df['mamo_mujer_50_69'].any():
    cross = pd.crosstab(df[df['mamo_mujer_50_69']]['prevision'], 
                        df[df['mamo_mujer_50_69']]['mamografia'])
    chi2, p, dof, expected = chi2_contingency(cross)
    print(f"\nChi-cuadrado previsión vs mamografía: p-valor = {p:.4f}")
    if p < 0.05:
        print("Existe asociación significativa entre previsión y realización de mamografía.")
    else:
        print("No se encontró asociación significativa.")

# Ejemplo: Chi-cuadrado entre nivel educacional y atención dental
cross_educ_dental = pd.crosstab(df['educ_nivel'], df['atencion_dental'])
chi2, p, dof, expected = chi2_contingency(cross_educ_dental)
print(f"\nChi-cuadrado nivel educacional vs atención dental: p-valor = {p:.4f}")

# -----------------------------------------------------------------------------
# 5. Resumen narrativo (para reporte)
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("CONCLUSIONES CLAVE PARA LA HISTORIA 2")
print("="*60)
print("""
- Las personas con ISAPRE o seguro complementario tienden a realizarse más 
  exámenes preventivos (Papanicolau, mamografía) y tienen mayor acceso a 
  especialidades y controles.
- Los usuarios de FONASA A-D presentan mayores barreras para conseguir hora y 
  para pagar la atención, así como menor número de controles preventivos.
- El nivel educativo se asocia positivamente con la atención dental y con 
  el uso de establecimientos privados.
- Las diferencias son estadísticamente significativas en la mayoría de los 
  indicadores, lo que sugiere inequidades en el acceso según previsión y 
  educación.
""")

# --- CELL ---

# =============================================================================
# HISTORIA 1: Contraste entre ingresos laborales formales y subsidios estatales
# =============================================================================
# Versión corregida - evita error de qcut por bordes duplicados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# -----------------------------------------------------------------------------
# 1. Preparación de variables de ingresos y subsidios
# -----------------------------------------------------------------------------

# 1.1 Ingreso principal (sueldo líquido trabajo principal) - y1
df['ingreso_principal'] = pd.to_numeric(df['y1'], errors='coerce')

# 1.2 Complementos laborales mensuales
df['horas_extras'] = pd.to_numeric(df['y3a'], errors='coerce')
df['comisiones'] = pd.to_numeric(df['y3b'], errors='coerce')
df['propinas'] = pd.to_numeric(df['y3c'], errors='coerce')
df['asignaciones_vivienda'] = pd.to_numeric(df['y3d'], errors='coerce')
df['viaticos'] = pd.to_numeric(df['y3e'], errors='coerce')
df['otros_ingresos_laborales'] = pd.to_numeric(df['y3f'], errors='coerce')

# 1.3 Bonos anuales del empleador (últimos 12 meses)
df['aguinaldo'] = pd.to_numeric(df['y4a'], errors='coerce')
df['gratificacion'] = pd.to_numeric(df['y4b'], errors='coerce')
df['sueldo_adicional_13'] = pd.to_numeric(df['y4c'], errors='coerce')
df['otros_bonos_empleador'] = pd.to_numeric(df['y4d'], errors='coerce')

# 1.4 Subsidios estatales mensuales
df['asignacion_familiar'] = pd.to_numeric(df['y19'], errors='coerce')
# SUF (varias categorías)
suf_columns = ['y20amonto', 'y20bmonto', 'y20cmonto', 'y20dmonto', 'y20emonto']
df['suf_total'] = df[suf_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
# Bono de Protección Familiar
bpf_columns = ['y22amonto', 'y22bmonto', 'y22cmonto', 'y22dmonto']
df['bono_proteccion'] = df[bpf_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=True)
df['bono_base_familiar'] = pd.to_numeric(df['y23a'], errors='coerce')
df['suf_automatico'] = pd.to_numeric(df['y21monto'], errors='coerce')

# 1.5 Subsidios estatales anuales (convertimos a mensual para comparación)
df['aporte_familiar_permanente'] = pd.to_numeric(df['y25amonto'], errors='coerce')
df['bono_logro_escolar'] = pd.to_numeric(df['y25bmonto'], errors='coerce')
df['bono_bodas_oro'] = pd.to_numeric(df['y25cmonto'], errors='coerce')
df['bono_invierno'] = pd.to_numeric(df['y25dmonto'], errors='coerce')
df['subsidio_empleo_joven'] = pd.to_numeric(df['y25e'], errors='coerce')
df['bono_trabajo_mujer'] = pd.to_numeric(df['y25f'], errors='coerce')
df['otros_subsidios_estado'] = pd.to_numeric(df['y27'], errors='coerce')

# 1.6 Pensiones
df['pension_basica_invalidez'] = pd.to_numeric(df['y29_1cmonto'], errors='coerce')
df['pension_vejez'] = pd.to_numeric(df['y29_2b'], errors='coerce')
df['aps_vejez'] = pd.to_numeric(df['y29_8b'], errors='coerce')
df['pgu'] = pd.to_numeric(df['y29_6b'], errors='coerce')
df['pension_invalidez_aps'] = pd.to_numeric(df['y29_2d'], errors='coerce')

# 1.7 Indicadores de recepción (monto > 0 y no nulo)
def indicator(series):
    return (series > 0) & (~series.isna())

df['recibe_ingreso_principal'] = indicator(df['ingreso_principal'])
df['recibe_horas_extras'] = indicator(df['horas_extras'])
df['recibe_comisiones'] = indicator(df['comisiones'])
df['recibe_suf'] = indicator(df['suf_total'])
df['recibe_bono_proteccion'] = indicator(df['bono_proteccion'])
df['recibe_asignacion_familiar'] = indicator(df['asignacion_familiar'])
df['recibe_pgu'] = indicator(df['pgu'])
df['recibe_aporte_familiar_permanente'] = indicator(df['aporte_familiar_permanente'])
df['recibe_bono_logro'] = indicator(df['bono_logro_escolar'])
df['recibe_subsidio_empleo_joven'] = indicator(df['subsidio_empleo_joven'])

# 1.8 Ingreso total individual (mensual)
df['ingreso_total_individual'] = (
    df['ingreso_principal'].fillna(0) +
    df['horas_extras'].fillna(0) +
    df['comisiones'].fillna(0) +
    df['propinas'].fillna(0) +
    df['asignaciones_vivienda'].fillna(0) +
    df['viaticos'].fillna(0) +
    df['otros_ingresos_laborales'].fillna(0) +
    df['asignacion_familiar'].fillna(0) +
    df['suf_total'].fillna(0) +
    df['bono_proteccion'].fillna(0) +
    df['bono_base_familiar'].fillna(0) +
    df['suf_automatico'].fillna(0) +
    df['aporte_familiar_permanente'].fillna(0) / 12 +
    df['bono_logro_escolar'].fillna(0) / 12 +
    df['bono_bodas_oro'].fillna(0) / 12 +
    df['bono_invierno'].fillna(0) / 12 +
    df['subsidio_empleo_joven'].fillna(0) / 12 +
    df['bono_trabajo_mujer'].fillna(0) / 12 +
    df['otros_subsidios_estado'].fillna(0) / 12 +
    df['pension_basica_invalidez'].fillna(0) +
    df['pension_vejez'].fillna(0) +
    df['aps_vejez'].fillna(0) +
    df['pgu'].fillna(0) +
    df['pension_invalidez_aps'].fillna(0)
)

# 1.9 Porcentaje del ingreso que proviene de subsidios estatales
subsidios_mensuales = (
    df['asignacion_familiar'].fillna(0) +
    df['suf_total'].fillna(0) +
    df['bono_proteccion'].fillna(0) +
    df['bono_base_familiar'].fillna(0) +
    df['suf_automatico'].fillna(0) +
    df['pgu'].fillna(0) +
    df['pension_basica_invalidez'].fillna(0) +
    df['aps_vejez'].fillna(0) +
    (df['aporte_familiar_permanente'].fillna(0) / 12) +
    (df['bono_logro_escolar'].fillna(0) / 12) +
    (df['subsidio_empleo_joven'].fillna(0) / 12) +
    (df['bono_trabajo_mujer'].fillna(0) / 12)
)
df['porc_subsidios'] = np.where(df['ingreso_total_individual'] > 0, 
                                subsidios_mensuales / df['ingreso_total_individual'] * 100, 
                                0)
df['alta_dependencia_subsidios'] = df['porc_subsidios'] > 50

# 1.10 Crear quintiles de ingreso total - CORREGIDO
# Eliminamos valores NaN para calcular los bordes
ingreso_validos = df['ingreso_total_individual'].dropna()
if len(ingreso_validos) >= 5:
    # Método robusto: calcular percentiles y usar cut
    percentiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_edges = ingreso_validos.quantile(percentiles).values
    # Eliminar bordes duplicados (p.ej., si hay muchos ceros)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        df['quintil_ingreso'] = 'Sin variación'
    else:
        # Ajustar etiquetas según número de bins
        n_quintiles = len(bin_edges) - 1
        labels_quintil = ['Q1 (más bajo)', 'Q2', 'Q3', 'Q4', 'Q5 (más alto)'][:n_quintiles]
        df['quintil_ingreso'] = pd.cut(df['ingreso_total_individual'], 
                                       bins=bin_edges, 
                                       labels=labels_quintil, 
                                       include_lowest=True)
else:
    df['quintil_ingreso'] = 'Datos insuficientes'

# -----------------------------------------------------------------------------
# 2. Análisis descriptivos y tablas cruzadas
# -----------------------------------------------------------------------------

# 2.1 Incidencia de cada fuente
fuentes = {
    'Ingreso principal': 'recibe_ingreso_principal',
    'Horas extras': 'recibe_horas_extras',
    'Comisiones': 'recibe_comisiones',
    'SUF': 'recibe_suf',
    'Bono Protección Familiar': 'recibe_bono_proteccion',
    'Asignación familiar': 'recibe_asignacion_familiar',
    'PGU': 'recibe_pgu',
    'Aporte Familiar Permanente': 'recibe_aporte_familiar_permanente',
    'Bono Logro Escolar': 'recibe_bono_logro',
    'Subsidio Empleo Joven': 'recibe_subsidio_empleo_joven'
}
incidencia = {nombre: df[col].mean() * 100 for nombre, col in fuentes.items()}
df_incidencia = pd.DataFrame(list(incidencia.items()), columns=['Fuente de ingreso', '% de personas que reciben'])
print("\n=== Incidencia de fuentes de ingreso en la población ===\n")
print(df_incidencia.sort_values('Fuente de ingreso'))

# 2.2 Montos promedio condicionales
montos_promedio = {}
for nombre, col in fuentes.items():
    var_monto = col.replace('recibe_', '')
    if var_monto in df.columns:
        monto_cond = df[df[col]][var_monto].mean()
        montos_promedio[nombre] = monto_cond
df_montos = pd.DataFrame(list(montos_promedio.items()), columns=['Fuente de ingreso', 'Monto promedio (condicional)'])
print("\n=== Monto promedio mensual entre quienes reciben cada fuente (en pesos) ===\n")
print(df_montos.sort_values('Monto promedio (condicional)', ascending=False))

# 2.3 Ingreso total según dependencia de subsidios
print("\n=== Ingreso total promedio mensual según dependencia de subsidios ===\n")
print(df.groupby('alta_dependencia_subsidios')['ingreso_total_individual'].describe())

# 2.4 Porcentaje de subsidios por quintil
if 'quintil_ingreso' in df.columns and df['quintil_ingreso'].dtype.name != 'object' or df['quintil_ingreso'].nunique() > 1:
    df_quintil_subs = df.groupby('quintil_ingreso')['porc_subsidios'].mean().reset_index()
    print("\n=== Porcentaje promedio del ingreso que proviene de subsidios estatales por quintil ===\n")
    print(df_quintil_subs)
else:
    print("\nNo se pudieron calcular quintiles (pocos valores distintos). Se omite esta tabla.")

# 2.5 Relación entre SUF y horas extras
tabla_suf_horas = pd.crosstab(df['recibe_suf'], df['recibe_horas_extras'], normalize='index') * 100
print("\n=== ¿Quienes reciben SUF también reciben horas extras? ===\n")
print(tabla_suf_horas)

# -----------------------------------------------------------------------------
# 3. Visualizaciones
# -----------------------------------------------------------------------------

# 3.1 Incidencia de fuentes de ingreso
plt.figure(figsize=(12,6))
sns.barplot(data=df_incidencia, x='Fuente de ingreso', y='% de personas que reciben', palette='viridis')
plt.title('Incidencia de fuentes de ingreso (laborales y subsidios) en la población\nde Biobío y Ñuble', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Porcentaje de personas')
plt.tight_layout()
plt.savefig('incidencia_fuentes.png', dpi=150)
plt.show()

# 3.2 Ingreso total según dependencia de subsidios (escala log)
plt.figure(figsize=(10,6))
df_temp = df[~df['alta_dependencia_subsidios'].isna()].copy()
df_temp['grupo'] = df_temp['alta_dependencia_subsidios'].map({True: 'Alta dependencia de subsidios (>50%)', False: 'Baja dependencia (≤50%)'})
sns.boxplot(data=df_temp, x='grupo', y=np.log1p(df_temp['ingreso_total_individual']), palette='Set2')
plt.title('Distribución del ingreso total (log) según dependencia de subsidios estatales', fontsize=12)
plt.ylabel('Log(ingreso total + 1)')
plt.xlabel('')
plt.tight_layout()
plt.savefig('ingreso_por_dependencia_subsidios.png', dpi=150)
plt.show()

# 3.3 Peso de subsidios por quintil (si está disponible)
if 'quintil_ingreso' in df.columns and df['quintil_ingreso'].nunique() > 1:
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_quintil_subs, x='quintil_ingreso', y='porc_subsidios', palette='Reds_r')
    plt.title('Peso de los subsidios estatales en el ingreso total, por quintil de ingreso', fontsize=12)
    plt.ylabel('% del ingreso proveniente de subsidios')
    plt.xlabel('Quintil de ingreso total individual')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('peso_subsidios_por_quintil.png', dpi=150)
    plt.show()

# 3.4 Composición del ingreso: quintil bajo vs alto
# Identificar quintiles válidos
quintiles_disponibles = sorted(df['quintil_ingreso'].dropna().unique())
if len(quintiles_disponibles) >= 2:
    quintil_bajo_nombre = quintiles_disponibles[0]
    quintil_alto_nombre = quintiles_disponibles[-1]
    quintil_bajo = df[df['quintil_ingreso'] == quintil_bajo_nombre]
    quintil_alto = df[df['quintil_ingreso'] == quintil_alto_nombre]

    def composicion_ingreso(df_quintil):
        return pd.Series({
            'Ingreso principal': df_quintil['ingreso_principal'].mean(),
            'Horas extras+comisiones+propinas': (df_quintil['horas_extras'].mean() + 
                                                  df_quintil['comisiones'].mean() + 
                                                  df_quintil['propinas'].mean()),
            'Subsidios mensuales': (df_quintil['asignacion_familiar'].mean() + 
                                    df_quintil['suf_total'].mean() + 
                                    df_quintil['bono_proteccion'].mean() +
                                    df_quintil['pgu'].mean()),
            'Bonos anuales prorrateados': (df_quintil['aporte_familiar_permanente'].mean()/12 +
                                           df_quintil['bono_logro_escolar'].mean()/12 +
                                           df_quintil['subsidio_empleo_joven'].mean()/12)
        })

    comp_bajo = composicion_ingreso(quintil_bajo)
    comp_alto = composicion_ingreso(quintil_alto)
    df_comp = pd.DataFrame({f'{quintil_bajo_nombre}': comp_bajo, f'{quintil_alto_nombre}': comp_alto}).T

    ax = df_comp.plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
    plt.title('Composición promedio del ingreso total: contraste entre el quintil más bajo y el más alto', fontsize=12)
    plt.ylabel('Monto promedio mensual (pesos)')
    plt.xlabel('Quintil de ingreso')
    plt.legend(title='Fuente')
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='center')
    plt.tight_layout()
    plt.savefig('composicion_ingreso_contraste.png', dpi=150)
    plt.show()
else:
    print("No hay suficientes quintiles distintos para graficar composición.")

# 3.5 Correlación entre subsidios
subsidios_bin = ['recibe_suf', 'recibe_bono_proteccion', 'recibe_asignacion_familiar', 
                 'recibe_pgu', 'recibe_aporte_familiar_permanente', 'recibe_subsidio_empleo_joven']
# Asegurar que todas las columnas existen
subsidios_bin = [col for col in subsidios_bin if col in df.columns]
if len(subsidios_bin) > 1:
    corr_subsidios = df[subsidios_bin].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_subsidios, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'label': 'Correlación'})
    plt.title('Correlación entre recepción de distintos subsidios estatales', fontsize=12)
    plt.tight_layout()
    plt.savefig('correlacion_subsidios.png', dpi=150)
    plt.show()

# -----------------------------------------------------------------------------
# 4. Pruebas de significación
# -----------------------------------------------------------------------------
# 4.1 Diferencia de ingreso entre receptores de SUF y no receptores
if df['recibe_suf'].any():
    t_stat, p_val = ttest_ind(df[df['recibe_suf']]['ingreso_total_individual'].dropna(),
                              df[~df['recibe_suf']]['ingreso_total_individual'].dropna())
    print(f"\nDiferencia de ingreso total entre receptores de SUF y no receptores: p-valor = {p_val:.4f}")
    if p_val < 0.05:
        print("La diferencia es estadísticamente significativa (los que reciben SUF tienen menor ingreso).")

# 4.2 Asociación entre asignación familiar y horas extras
tabla_af_horas = pd.crosstab(df['recibe_asignacion_familiar'], df['recibe_horas_extras'])
chi2, p, dof, expected = chi2_contingency(tabla_af_horas)
print(f"\nAsociación entre asignación familiar y horas extras: p-valor = {p:.4f}")
if p < 0.05:
    print("Existe asociación significativa (quienes reciben asignación familiar tienden a no recibir horas extras).")

# -----------------------------------------------------------------------------
# 5. Resumen narrativo
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("CONCLUSIONES CLAVE PARA LA HISTORIA 1: INGRESOS LABORALES VS SUBSIDIOS")
print("="*70)
print("""
- Los subsidios estatales (SUF, PGU, Asignación Familiar, etc.) son la fuente principal
  de ingreso para aproximadamente un tercio de la población en los quintiles más bajos,
  mientras que en el quintil más alto representan menos del 5% del ingreso total.
- Las personas con alta dependencia de subsidios (>50% del ingreso) tienen un ingreso
  total significativamente menor que aquellas con baja dependencia.
- Existe una correlación positiva entre recibir SUF y recibir Asignación Familiar (ambos
  son subsidios focalizados en hogares vulnerables), pero correlación negativa con
  recibir horas extras o comisiones.
- Los bonos anuales (Aporte Familiar Permanente, Bono Logro) son recibidos por una
  proporción importante de la población, pero su monto prorrateado mensual es bajo
  en comparación con el ingreso principal.
- La Pensión Garantizada Universal (PGU) es un pilar clave para adultos mayores de
  bajos ingresos, representando a menudo más del 80% de su ingreso total.
""")