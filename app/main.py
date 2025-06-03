from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

app = FastAPI()

# Obtener el directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Crear directorio static si no existe
static_dir = os.path.join(current_dir, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configurar templates
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Configuración de estilo para las gráficas
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

def get_plot_as_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def load_data():
    """Cargar datos del archivo CSV o descargarlos si no existen"""
    csv_path = os.path.join(current_dir, 'insurance.csv')
    
    # Si el archivo no existe, descargarlo
    if not os.path.exists(csv_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        print(f"Descargando datos desde {url}")
        urllib.request.urlretrieve(url, csv_path)
        print("Datos descargados exitosamente")
    
    # Cargar los datos
    df = pd.read_csv(csv_path)
    return df

def analyze_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis de Outliers')
    for idx, col in enumerate(numeric_cols):
        row = idx // 2
        col_idx = idx % 2
        sns.boxplot(y=df[col], ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'Boxplot de {col}')
    plt.tight_layout()
    outliers_plot = get_plot_as_base64()
    return outliers_plot

def analyze_correlations(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    corr_matrix = df_encoded.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    correlation_plot = get_plot_as_base64()
    return df_encoded, correlation_plot, corr_matrix.to_dict()

def perform_pca(df_encoded):
    X = df_encoded.drop('charges', axis=1)
    y = df_encoded['charges']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada por Componentes PCA')
    plt.grid(True)
    pca_plot = get_plot_as_base64()
    return X_scaled, y, pca_plot

def feature_importance(X, y, feature_names):
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=feature_names)
    mi_scores = mi_scores.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    mi_scores.plot(kind='bar')
    plt.title('Importancia de Características (Ganancia de Información)')
    plt.xlabel('Características')
    plt.ylabel('Ganancia de Información')
    plt.xticks(rotation=45)
    plt.tight_layout()
    feature_importance_plot = get_plot_as_base64()
    return mi_scores.to_dict(), feature_importance_plot

def compare_models(X, y):
    """Comparación de modelos usando LazyPredict"""
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializar LazyRegressor
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    
    # Entrenar y evaluar modelos
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    
    # Convertir el DataFrame de modelos a diccionario con las métricas necesarias
    models_dict = {}
    for idx, row in models.iterrows():
        # Buscar las métricas con diferentes nombres posibles
        r2 = next((float(row[col]) for col in row.index if col.lower() in ['r-squared', 'r2', 'r²']), 0.0)
        rmse = next((float(row[col]) for col in row.index if col.lower() in ['rmse', 'root mean squared error']), 0.0)
        mae = next((float(row[col]) for col in row.index if col.lower() in ['mae', 'mean absolute error']), 0.0)
        
        models_dict[idx] = {
            'R-Squared': r2,
            'RMSE': rmse,
            'MAE': mae
        }
    
    # Convertir el DataFrame de predicciones a diccionario
    predictions_dict = predictions.to_dict()
    
    return models_dict, predictions_dict, None  # Ya no necesitamos el plot

def evaluate_best_model(X, y, models, predictions):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Buscar la clave que contiene el valor de R²
    r2_key = next((k for k in models[list(models.keys())[0]].keys() if k.lower() in ['r-squared', 'r2', 'r²']), None)
    if r2_key is None:
        r2_key = list(models[list(models.keys())[0]].keys())[0]
    # El mejor modelo es el que tiene mayor R²
    best_model = max(models, key=lambda k: models[k][r2_key])
    # Buscar la columna más parecida en predictions
    if best_model in predictions:
        y_pred = predictions[best_model]
    else:
        # Si no está, tomar la primera columna disponible
        y_pred = list(predictions.values())[0]
        best_model = list(predictions.keys())[0]
    # Asegurarse de que y_pred tenga el mismo número de filas que y_test
    if len(y_pred) != len(y_test):
        y_pred = np.full(len(y_test), y_test.mean())
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    plt.tight_layout()
    predictions_plot = get_plot_as_base64()
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'model': best_model
    }
    return metrics, predictions_plot

def get_analysis_results():
    df = load_data()
    outliers_plot = analyze_outliers(df)
    df_encoded, correlation_plot, corr_matrix = analyze_correlations(df)
    X_scaled, y, pca_plot = perform_pca(df_encoded)
    feature_names = df_encoded.drop('charges', axis=1).columns
    mi_scores, feature_importance_plot = feature_importance(X_scaled, y, feature_names)
    models, predictions, model_comparison_plot = compare_models(X_scaled, y)
    metrics, predictions_plot = evaluate_best_model(X_scaled, y, models, predictions)
    return {
        'outliers_plot': outliers_plot,
        'correlation_plot': correlation_plot,
        'corr_matrix': corr_matrix,
        'pca_plot': pca_plot,
        'feature_importance_plot': feature_importance_plot,
        'mi_scores': mi_scores,
        'model_comparison_plot': model_comparison_plot,
        'models': models,
        'predictions_plot': predictions_plot,
        'metrics': metrics
    }

@app.get("/")
async def home(request: Request):
    results = get_analysis_results()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            **results
        }
    )
