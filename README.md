# Análisis de Costos Médicos con Machine Learning

Este proyecto implementa un análisis completo de costos médicos utilizando técnicas de Machine Learning y visualización de datos. La aplicación está construida con FastAPI y utiliza Docker para su despliegue.

## 🚀 Tecnologías Utilizadas

### Backend
- **Python 3.9**: Lenguaje principal de programación
- **FastAPI**: Framework web moderno y rápido para APIs
- **Uvicorn**: Servidor ASGI para ejecutar la aplicación

### Análisis de Datos y Machine Learning
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Computación numérica
- **Scikit-learn**: Machine Learning y procesamiento de datos
- **LazyPredict**: Automatización de comparación de modelos
- **Matplotlib & Seaborn**: Visualización de datos

### Frontend
- **Jinja2**: Motor de plantillas para renderizar HTML
- **Bootstrap 5**: Framework CSS para el diseño responsive

### DevOps
- **Docker**: Contenerización de la aplicación
- **Docker Compose**: Orquestación de contenedores

## 📋 Prerrequisitos

- Docker
- Docker Compose
- Git (opcional, para clonar el repositorio)

## 🛠️ Instalación y Ejecución

1. **Clonar el repositorio** (opcional):
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. **Construir la imagen de Docker**:
```bash
docker-compose build
```

3. **Iniciar la aplicación**:
```bash
docker-compose up -d
```

4. **Verificar que la aplicación está corriendo**:
```bash
docker-compose ps
```

La aplicación estará disponible en: `http://localhost:8000`

## 📊 Características del Análisis

1. **Análisis Exploratorio de Datos (EDA)**:
   - Detección de outliers
   - Matriz de correlación
   - Análisis de componentes principales (PCA)

2. **Selección de Características**:
   - Importancia de características usando ganancia de información
   - Visualización de la importancia de cada variable

3. **Comparación de Modelos**:
   - Evaluación automática de múltiples modelos de regresión
   - Métricas de rendimiento: R², RMSE, MAE
   - Visualización de predicciones vs valores reales

## 🐳 Comandos Docker Útiles

- **Ver logs en tiempo real**:
```bash
docker-compose logs -f
```

- **Detener la aplicación**:
```bash
docker-compose down
```

- **Reiniciar la aplicación**:
```bash
docker-compose restart
```

- **Reconstruir y reiniciar** (después de cambios):
```bash
docker-compose up -d --build
```

## 📁 Estructura del Proyecto

```
.
├── app/
│   ├── main.py           # Aplicación principal FastAPI
│   ├── static/          # Archivos estáticos
│   └── templates/       # Plantillas HTML
├── Dockerfile           # Configuración de la imagen Docker
├── docker-compose.yml   # Configuración de servicios Docker
├── requirements.txt     # Dependencias de Python
└── README.md           # Este archivo
```

## 🔧 Configuración del Entorno

El proyecto utiliza las siguientes versiones específicas de las dependencias:
- FastAPI 0.104.1
- Uvicorn 0.24.0
- Pandas 2.1.3
- NumPy 1.26.2
- Scikit-learn 1.3.2
- LazyPredict 0.2.12

## 📝 Notas Adicionales

- La aplicación está configurada para reiniciarse automáticamente en caso de fallo (`restart: unless-stopped`)
- Los cambios en el código se reflejan automáticamente gracias al volumen montado en Docker
- El puerto 8000 debe estar disponible en tu máquina local

## 🤝 Contribución

Si deseas contribuir al proyecto:
1. Haz un Fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles. 