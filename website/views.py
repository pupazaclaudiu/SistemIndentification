import os
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, flash, current_app,redirect, url_for
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

views = Blueprint('views', __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

METHODS=[
    {"id":1,"name":"Semnale (procese) stochastice (aleatoare)","endpoint":"l1"},
    {"id":2,"name":"Filtrarea numerică a datelor","endpoint":"l2"},
    {"id":3,"name":"Identificarea sistemelor utilizând metode de regresie","endpoint":"l3"},
    {"id":4,"name":"Identificarea sistemelor prin metoda deconvoluţiei","endpoint":"l4"},
    {"id":5,"name":"CMMP offline","endpoint":"l5"},
    {"id":6,"name":"Estimatori CMMP recursivi (on-line)","endpoint":"l6"},
    {"id":7,"name":"Curve fitting ","endpoint":"l7"},
    {"id":8,"name":"Neural Network","endpoint":"l8"}
]

@views.route('/')
def home():
    return render_template('home.html')

@views.route('/upload', methods=['GET', 'POST'])
def upload():
    active_method = None
    current_file_name = None

    if request.method == 'POST':
        # 1) citim metoda și fișierul
        method_id = request.form.get('method_id')
        file = request.files.get('csv_file')

        has_error = False

        # verificare metodă
        if not method_id:
            flash('Te rog selectează o metoda înainte de a porni analiza.', 'error')
            has_error = True
        else:
            active_method = int(method_id)

        # verificare fișier
        if not file or file.filename == '':
            flash('Te rog încarcă un fișier .csv înainte de a porni analiza.', 'error')
            has_error = True
        else:
            current_file_name = file.filename

        if has_error:
            # reafișăm pagina cu erorile și cu selectul setat corect
            return render_template(
                'mainbase.html',
                methods=METHODS,
                active_method=active_method,
                current_file_name=current_file_name,
            )
        # 🟢 dacă e totul ok:
        # 1. salvăm fișierul
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # 2. construim endpoint-ul laboratorului
        endpoint_name = f"l{method_id}"   # l1, l2, ...

        # 3. redirect către laborator, trimițând numele fișierului în URL
        return redirect(url_for(f'views.{endpoint_name}', filename=filename))

    # GET simplu: doar afișăm pagina
    return render_template(
        'mainbase.html',
        methods=METHODS,
        active_method=active_method,
        current_file_name=current_file_name,
    )


@views.route('/l1', methods=['GET'])
def l1():
    current_file_name = None
    n_rows = None
    n_cols = None

    stats_x_m = stats_x_v = stats_x_sigma = None
    stats_y_m = stats_y_v = stats_y_sigma = None

    corr_rxy0 = None
    corr_max_rxx = None
    corr_max_ryy = None

    plot_x_url = plot_y_url = plot_rxx_url = plot_rxy_url = None

    # luăm numele fișierului din /l1?filename=...
    filename = request.args.get('filename')

    if filename:
        current_file_name = filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        #Ghiceste el separatorul
        df = pd.read_csv(filepath, sep=None, engine="python")

        # dacă tot are mai puțin de 2 coloane, dăm eroare frumoasă
        if df.shape[1] < 2:
            flash('Fișierul trebuie să aibă cel puțin 2 coloane (X și Y).', 'error')
            return render_template(
                'l1.html',
                methods=METHODS,
                active_method=1,
                current_file_name=os.path.basename(filepath),
                n_rows=df.shape[0],
                n_cols=df.shape[1],
            )

        # X = prima coloană, Y = a doua coloană
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        n_rows, n_cols = df.shape

        # === STATISTICI X ===
        stats_x_m = float(x.mean())
        stats_x_v = float(x.var())
        stats_x_sigma = float(x.std())

        # === STATISTICI Y ===
        stats_y_m = float(y.mean())
        stats_y_v = float(y.var())
        stats_y_sigma = float(y.std())

        # === CORELATII (simplu exemplu) ===
        rxx = np.correlate(x - stats_x_m, x - stats_x_m, mode="full")
        ryy = np.correlate(y - stats_y_m, y - stats_y_m, mode="full")
        rxy = np.correlate(x - stats_x_m, y - stats_y_m, mode="full")

        corr_rxy0 = float(rxy[len(rxy)//2])
        corr_max_rxx = float(rxx.max())
        corr_max_ryy = float(ryy.max())

        # === GENERARE PLOT-URI ===
        plots_dir = os.path.join(current_app.static_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # X(t)
        plt.figure()
        plt.plot(x.values)
        plt.title("X în funcție de timp")
        plt.xlabel("n")
        plt.ylabel("X[n]")
        x_path = os.path.join(plots_dir, 'l1_x.png')
        plt.savefig(x_path, bbox_inches='tight')
        plt.close()

        # Y(t)
        plt.figure()
        plt.plot(y.values)
        plt.title("Y în funcție de timp")
        plt.xlabel("n")
        plt.ylabel("Y[n]")
        y_path = os.path.join(plots_dir, 'l1_y.png')
        plt.savefig(y_path, bbox_inches='tight')
        plt.close()

        # Rxx
        plt.figure()
        plt.plot(rxx)
        plt.title("Autocorelație Rxx")
        rxx_path = os.path.join(plots_dir, 'l1_rxx.png')
        plt.savefig(rxx_path, bbox_inches='tight')
        plt.close()

        # Rxy
        plt.figure()
        plt.plot(rxy)
        plt.title("Intercorelație Rxy")
        rxy_path = os.path.join(plots_dir, 'l1_rxy.png')
        plt.savefig(rxy_path, bbox_inches='tight')
        plt.close()

        # URL-urile către imagini (pentru <img src="..."> în template)
        plot_x_url = url_for('static', filename='plots/l1_x.png')
        plot_y_url = url_for('static', filename='plots/l1_y.png')
        plot_rxx_url = url_for('static', filename='plots/l1_rxx.png')
        plot_rxy_url = url_for('static', filename='plots/l1_rxy.png')

    return render_template(
        'l1.html',
        methods=METHODS,
        active_method=1,

        # info fișier
        current_file_name=current_file_name,
        n_rows=n_rows,
        n_cols=n_cols,

        # statistici X
        stats_x_m=stats_x_m,
        stats_x_v=stats_x_v,
        stats_x_sigma=stats_x_sigma,

        # statistici Y
        stats_y_m=stats_y_m,
        stats_y_v=stats_y_v,
        stats_y_sigma=stats_y_sigma,

        # corelații
        corr_rxy0=corr_rxy0,
        corr_max_rxx=corr_max_rxx,
        corr_max_ryy=corr_max_ryy,

        # plot-uri
        plot_x_url=plot_x_url,
        plot_y_url=plot_y_url,
        plot_rxx_url=plot_rxx_url,
        plot_rxy_url=plot_rxy_url,
    )
