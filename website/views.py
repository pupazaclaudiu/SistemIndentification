from flask import Blueprint,render_template,request, flash
import pandas as pd
views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template('home.html')

@views.route('/upload', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        # 1. luăm fișierul din formular
        file = request.files.get('file')

        if not file:
            flash("Nu ai selectat niciun fișier.",category="error")
            return render_template("upload.html")

        if file.filename == '':
            flash("Numele fișierului este gol.",category="error")
            return render_template("upload.html")

        # 2. verificăm extensia (simplu)
        if not file.filename.lower().endswith('.csv'):
            flash("Te rog încarcă un fișier cu extensia .csv",category="error")
            return render_template("upload.html")

            # 3. citim CSV-ul direct din obiectul fișier
        try:
            df = pd.read_csv(file)  # file este un file-like object
        except Exception as e:
            flash(f"Eroare la citirea CSV-ului: {e}")
            return render_template("upload.html")

            # 4. aici faci ANALIZA ta pe ele ex:

        head = df.head().to_html(classes="table table-striped", index=False)
        desc = df.describe(include='all').to_html(classes="table table-bordered")

            # poți trimite mai departe ce vrei în template
        return render_template("analysis.html", head=head, desc=desc)

     # dacă e GET, doar arătăm formularul
    return render_template("upload.html")