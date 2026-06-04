# Sanity check de embeddings

| Pareja | Texto A | Texto B | Coseno |
|---|---|---|---|
| A — cercanos (esperado > 0.6) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | 0.5957 |
| B — no relacionados (esperado < 0.4) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | 0.1920 |
| C — genéricos/ambiguos (sin expectativa fija) | Backend services | API development | 0.5406 |

## Comentario

Los resultados encajan con la intuición, con un matiz interesante en la pareja A.

- **Pareja A (0.5957):** ambos textos describen autenticación con JWT, pero el
  modelo se queda **justo por debajo** del umbral de 0.6 esperado. La distancia
  léxica (fintech mobile app vs banking application; OAuth 2.0 vs JSON Web
  Tokens) basta para no cruzar el 0.6. Confirma que `text-embedding-3-small`
  captura el solapamiento semántico pero no infla la similitud cuando el
  vocabulario de superficie difiere — útil tenerlo presente al fijar umbrales de
  retrieval en S08.
- **Pareja B (0.1920):** claramente por debajo de 0.4, como se esperaba. Auth
  con JWT y una migración de base de datos no comparten dominio; el modelo lo
  refleja con una similitud baja.
- **Pareja C (0.5406):** "Backend services" y "API development" son términos
  genéricos del mismo campo; una similitud media-alta sin expectativa fija es
  coherente. El valor relativamente alto sugiere que los textos cortos y
  abstractos tienden a agruparse, algo a vigilar para no recuperar ruido.
