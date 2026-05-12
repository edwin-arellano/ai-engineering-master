"""Ejemplos estáticos de estimaciones previas inyectados como contexto CAG.

Estos ejemplos son el "conocimiento" del sistema. En arquitectura CAG no
hay base de datos ni retrieval: todo el contexto viaja en cada llamada
al LLM como parte del system prompt.

Los datos son ficticios pero realistas, diseñados para que el modelo
calibre escala, formato y nivel de detalle de las estimaciones que debe
generar.
"""

ESTIMATION_EXAMPLES: list[dict[str, str]] = [
    {
        "meeting_summary": (
            "Plataforma web de gestión de inventario para una cadena de "
            "tiendas de retail medianas. Necesita CRUD de productos, "
            "control de stock multi-almacén, dashboard de métricas y "
            "autenticación con roles."
        ),
        "estimation": """## Estimación: Plataforma de Gestión de Inventario

### Desglose de tareas
1. Diseño UI/UX (40 h)
2. Backend API REST (CRUD inventario, stock, almacenes) (60 h)
3. Autenticación y gestión de roles (20 h)
4. Dashboard con métricas y reportes (30 h)
5. Testing y QA (25 h)

**Total: 175 h**
**Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 6-8 semanas**
""",
    },
    {
        "meeting_summary": (
            "App móvil para gestionar mantenimientos preventivos en flotas "
            "de vehículos comerciales. Login social, captura de fotos con "
            "geolocalización, sincronización offline, panel admin web para "
            "gestores."
        ),
        "estimation": """## Estimación: App Móvil de Mantenimiento de Flotas

### Desglose de tareas
1. Diseño UI/UX (mobile + admin web) (50 h)
2. App móvil React Native (auth social, captura fotos, geoloc) (90 h)
3. Sincronización offline con cola de eventos (35 h)
4. Backend API (vehículos, mantenimientos, sync) (70 h)
5. Panel admin web (40 h)
6. Testing en dispositivos reales y QA (35 h)

**Total: 320 h**
**Equipo recomendado: 1 mobile dev senior + 2 full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 10-12 semanas**
""",
    },
    {
        "meeting_summary": (
            "Integración entre un ERP propietario (SAP Business One) y una "
            "tienda Shopify. Sincronización bidireccional de productos, "
            "stock y pedidos. Webhooks, cola de procesamiento, panel de "
            "monitorización."
        ),
        "estimation": """## Estimación: Integración SAP B1 ↔ Shopify

### Desglose de tareas
1. Análisis del modelo de datos en ambos lados (15 h)
2. Servicio de sincronización (Node.js o Python) con cola Redis (60 h)
3. Webhooks Shopify + polling SAP B1 (25 h)
4. Mapeo y normalización de datos (productos, stock, pedidos) (30 h)
5. Panel de monitorización (jobs, errores, reintentos) (25 h)
6. Testing de integración y QA (20 h)

**Total: 175 h**
**Equipo recomendado: 1 backend senior con experiencia en integraciones + 1 desarrollador full-stack**
**Duración estimada: 6-7 semanas**
""",
    },
]
