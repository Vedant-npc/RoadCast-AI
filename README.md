# ROADCAST AI

AI-Powered Urban Traffic Congestion Nowcasting System for Pune, India.

## Phase 1: project setup

This phase focuses on creating a clean structure for the backend API, frontend client, and shared project configuration. The goal is to make sure the project is easy to run, extend, and debug before we start building the ML pipeline.

## Project structure

- backend/: FastAPI API service
- frontend/: Vite + React + Tailwind dashboard
- .env.example: shared environment example
- .gitignore: local environment protection

## Quick start

1. Create a Python virtual environment.
2. Install backend dependencies.
3. Install frontend dependencies.
4. Run the backend and frontend servers.

## Environment variables

Copy the example files and update them with your own local secrets.

## Backend setup and server

Create a Python virtual environment and install the backend dependencies:

```powershell
Set-Location "D:\My_Projects\RoadCast AI\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the API server:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check the API:

```powershell
curl http://localhost:8000/
curl http://localhost:8000/api/health
```

## Phase 7.3 traffic collector

The collector stores TomTom observations for five verified Pune query locations. These locations are external road query points and are not claimed to be METR-LA sensors. The API key remains in the server environment and is never written to SQLite.

From the repository root, initialize and collect one cycle:

```powershell
Set-Location "D:\My_Projects\RoadCast AI\backend"
.\.venv\Scripts\python.exe collect_traffic.py --once
```

Use `--dry-run` to query the configured locations without creating or writing the database:

```powershell
.\.venv\Scripts\python.exe collect_traffic.py --once --dry-run
```

Run continuous polling only when explicitly needed:

```powershell
.\.venv\Scripts\python.exe collect_traffic.py
```

Check the database:

```powershell
.\.venv\Scripts\python.exe collect_traffic.py --status
```

The default database is `backend/data/traffic_history.sqlite3`. Set `TRAFFIC_POLL_INTERVAL_SECONDS` to change the interval in seconds. Set `TRAFFIC_LOCATIONS_JSON` to replace the five-location registry with a JSON list containing `location_id`, `name`, `latitude`, and `longitude`. Do not use METR-LA node IDs or invented coordinates.
