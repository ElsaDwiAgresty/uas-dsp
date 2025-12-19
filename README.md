<<<<<<< HEAD
# uas-dsp
=======
# DSP Project

## Deskripsi

Proyek ini adalah aplikasi Flask untuk Digital Signal Processing.

## Persyaratan

- Python 3.11+
- Flask
- Numpy
- Scipy

## Instalasi

### Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

### Local Development

```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000`

### Docker

```bash
docker build -t dsp-app .
docker run -p 5000:5000 dsp-app
```

## Struktur Folder

```
dsp/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Procfile
├── README.md
├── app.py
├── requirements.txt
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── home.html
    └── dashboard_view.html
```
>>>>>>> 3037a8f (commit projek)
