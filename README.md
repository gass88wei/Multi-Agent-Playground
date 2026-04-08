# Multi-Agent Playground

This repo is now self-contained at:

`C:\Users\zhang\MySpace\study\dev\ai\Multi-Agent-Playground`

## Runtime layout

- Python venv: `.\backend\.venv`
- Env file: `.\.env`
- Backend: `.\backend`
- Frontend: `.\frontend`

Backend config is loaded from the project root `.env` only.

## Quick setup (Windows, no PowerShell scripts)

```powershell
cd C:\Users\zhang\MySpace\study\dev\ai\Multi-Agent-Playground
C:\Users\zhang\AppData\Local\Programs\Python\Python312\python.exe -m venv .\backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install --upgrade pip
.\backend\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

Then configure:

```text
.\.env
```

At minimum, set `OPENAI_API_KEY`.

## Start services

Backend:

```powershell
cd .\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8011 --reload
```

Frontend:

```powershell
cd .\frontend
npm run dev
```

Frontend proxy default target is `http://127.0.0.1:8011`.

## Notes

- Use Python from `.\backend\.venv\Scripts\python.exe`.
- If `.\.env` does not exist, copy from `.\.env.example`.
- Backend dependencies are in `.\backend\requirements.txt`.
