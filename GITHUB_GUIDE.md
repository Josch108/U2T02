# Guía para Conectar y Subir el Repositorio a GitHub

Esta guía describe paso a paso cómo vincular este proyecto local (`U2T2`) con un nuevo repositorio en tu cuenta de GitHub y subir todos los archivos de la entrega.

---

## Estado Actual del Repositorio Local

El repositorio local ya fue inicializado con la rama principal `main` y cuenta con un archivo `.gitignore` configurado para evitar subir archivos pesados (pesos `.pt`, `.safetensors`, directorios de checkpoints `runs/` y datasets `.jsonl`).

---

## Paso 1: Crear el Repositorio en GitHub

Puedes crear el repositorio remoto mediante cualquiera de las siguientes dos opciones:

### Opción A: Desde la Web de GitHub
1. Inicia sesión en [GitHub](https://github.com).
2. Haz clic en el botón **`+`** (arriba a la derecha) y selecciona **New repository**.
3. Asigna un nombre al repositorio (por ejemplo: `U2T02-SimCSE` o `simcse-sentence-embeddings`).
4. Configura la visibilidad en **Public** o **Private** (según las indicaciones de tu curso o equipo).
5. **IMPORTANTE:** Deja desmarcadas las casillas *Add a README file*, *Add .gitignore* y *Choose a license* (ya las tenemos creadas localmente).
6. Haz clic en **Create repository**.
7. Copia la URL del repositorio remoto (HTTPS o SSH), por ejemplo:
   - HTTPS: `https://github.com/TU_USUARIO/U2T02-SimCSE.git`
   - SSH: `git@github.com:TU_USUARIO/U2T02-SimCSE.git`

### Opción B: Con GitHub CLI (`gh`)
Si tienes instalado `gh` en tu terminal:
```bash
cd /Users/josue/Desktop/codiguitos/9no/trends_data_science/U2T2
gh repo create U2T02-SimCSE --public --source=. --remote=origin
```

---

## Paso 2: Realizar el Primer Commit Local

Abre tu terminal en la carpeta del proyecto y ejecuta:

```bash
cd /Users/josue/Desktop/codiguitos/9no/trends_data_science/U2T2

# 1. Agregar todos los archivos preparados
git add .

# 2. Verificar que los archivos pesados no se incluyan
git status

# 3. Crear el commit inicial
git commit -m "Initial commit: SimCSE replication pipeline, evaluation and report (U2T02)"
```

---

## Paso 3: Vincular el Remoto y Subir a GitHub

Si creaste el repositorio desde la web (Opción A), vincula la URL copiada y sube la rama `main`:

```bash
# Vincular el origen remoto (reemplaza con tu URL real)
git remote add origin https://github.com/TU_USUARIO/U2T02-SimCSE.git

# Asegurar que la rama se llame main
git branch -M main

# Subir los cambios
git push -u origin main
```

---

## Paso 4: Autenticación en Git (si te la solicita)

Si usas HTTPS y Git te solicita contraseña:
- GitHub no acepta contraseñas convencionales desde la terminal; requiere un **Personal Access Token (Classic o Fine-grained)**.
- Para generarlo: Ve a **Settings -> Developer Settings -> Personal access tokens -> Tokens (classic)**, crea uno con permisos de `repo` y úsalo como contraseña.
- Alternativamente, si usas SSH, asegúrate de que tu llave pública esté registrada en **Settings -> SSH and GPG keys**.

---

## Flujo de Trabajo Posterior (Subir Avances o Reportes)

Cada vez que actualices el reporte, ejecutes entrenamientos o agregues notas:

```bash
git add report/ notebooks/ src/ README.md
git commit -m "docs: actualizar reporte técnico y resultados de STS-B"
git push
```
