import os
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, flash, current_app,redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from scipy.signal import lfilter

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
        plots_dir = os.path.join(current_app.static_folder, 'plots/l1')
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
        plot_x_url = url_for('static', filename='plots/l1/l1_x.png')
        plot_y_url = url_for('static', filename='plots/l1/l1_y.png')
        plot_rxx_url = url_for('static', filename='plots/l1/l1_rxx.png')
        plot_rxy_url = url_for('static', filename='plots/l1/l1_rxy.png')

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
@views.route('/l2')
def l2():
    current_file_name = None
    n_rows = None
    n_cols = None

    plot_x_url = None
    plot_lp_url = None
    plot_hp_url = None
    plot_cornell_url = None

    # 6 imagini FFT separate
    plot_fft1_url = plot_fft2_url = None
    plot_fft3_url = plot_fft4_url = None
    plot_fft5_url = plot_fft6_url = None

    filename = request.args.get('filename')

    if filename:
        current_file_name = filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # citire CSV cu separator auto-detectat (',' sau ';')
        df = pd.read_csv(filepath, sep=None, engine="python")
        n_rows, n_cols = df.shape

        if n_cols < 1:
            flash('Fișierul trebuie să aibă cel puțin o coloană pentru L2.', 'error')
        else:
            # X = prima coloană
            x = df.iloc[:, 0].to_numpy()

            # ==========================
            # Filtre ordin I (LP, HP, Cornell)
            # ==========================
            T = 0.01
            alpha = 0.7
            t = np.arange(len(x)) * T

            # trece-jos
            b_lp = [1 - alpha, 0]
            a_lp = [1, -alpha]

            # trece-sus
            b_hp = [1 - alpha, 0]
            a_hp = [1, alpha]

            # Cornell (trece-sus îmbunătățit)
            b_cornell = [(1 - alpha) / 2, -(1 - alpha) / 2]
            a_cornell = [1, -(1 - alpha)]

            y_lp = lfilter(b_lp, a_lp, x)
            y_hp = lfilter(b_hp, a_hp, x)
            y_cornell = lfilter(b_cornell, a_cornell, x)

            # ==========================
            # Analiză FFT (semnal sintetic, cum era în exemplu)
            # ==========================
            N = 128
            i = np.arange(1, N + 1)
            q = np.sin(i * 14 * np.pi / 128) + np.cos(i * 19 * np.pi / 128)

            p = 2 * np.random.rand(N) - 1
            s = q + p

            fq = np.fft.fft(q)
            fs = np.fft.fft(s)

            alfa = 16
            g = np.zeros_like(fs, dtype=complex)
            for k in range(N):
                if np.abs(fs[k]) > alfa:
                    g[k] = fs[k]

            h = np.fft.ifft(g)
            H = np.fft.fft(h)

            # ==========================
            # Salvare plot-uri
            # ==========================
            plots_dir = os.path.join(current_app.static_folder, 'plots/l2')
            os.makedirs(plots_dir, exist_ok=True)

            # --- semnal original (din fișier) ---
            plt.figure()
            plt.plot(t, x)
            plt.title('L2 – Semnal original (din fișier)')
            plt.xlabel('t')
            plt.ylabel('x[t]')
            x_path = os.path.join(plots_dir, 'l2_x.png')
            plt.savefig(x_path, bbox_inches='tight')
            plt.close()

            # --- LP ---
            plt.figure()
            plt.plot(t, y_lp)
            plt.title('Filtru trece-jos')
            plt.xlabel('t')
            plt.ylabel('y_{LP}[t]')
            lp_path = os.path.join(plots_dir, 'l2_lp.png')
            plt.savefig(lp_path, bbox_inches='tight')
            plt.close()

            # --- HP ---
            plt.figure()
            plt.plot(t, y_hp)
            plt.title('Filtru trece-sus')
            plt.xlabel('t')
            plt.ylabel('y_{HP}[t]')
            hp_path = os.path.join(plots_dir, 'l2_hp.png')
            plt.savefig(hp_path, bbox_inches='tight')
            plt.close()

            # --- Cornell ---
            plt.figure()
            plt.plot(t, y_cornell)
            plt.title('Filtru Cornell (trece-sus)')
            plt.xlabel('t')
            plt.ylabel('y_{Cornell}[t]')
            cornell_path = os.path.join(plots_dir, 'l2_cornell.png')
            plt.savefig(cornell_path, bbox_inches='tight')
            plt.close()

            # ==========================
            # 6 FIGURI FFT INDIVIDUALE
            # ==========================

            # 1) Semnal util
            plt.figure(figsize=(8, 3))
            plt.plot(i, q, 'b')
            plt.title('1) Semnal util')
            plt.grid(True)
            fft1_path = os.path.join(plots_dir, 'l2_fft1.png')
            plt.savefig(fft1_path, bbox_inches='tight')
            plt.close()

            # 2) Perturbație + semnal perturbat
            plt.figure(figsize=(8, 3))
            plt.plot(i, p, 'r', label='Perturbație')
            plt.plot(i, s, 'k', label='Semnal perturbat', linewidth=0.8)
            plt.title('2) Perturbație și semnal perturbat')
            plt.legend()
            plt.grid(True)
            fft2_path = os.path.join(plots_dir, 'l2_fft2.png')
            plt.savefig(fft2_path, bbox_inches='tight')
            plt.close()

            # 3) Spectru util vs perturbat
            plt.figure(figsize=(8, 3))
            plt.plot(np.abs(fq), 'b', label='|FFT(q)| – util')
            plt.plot(np.abs(fs), 'r', label='|FFT(s)| – perturbat', alpha=0.7)
            plt.title('3) Spectru util vs perturbat')
            plt.legend()
            plt.grid(True)
            fft3_path = os.path.join(plots_dir, 'l2_fft3.png')
            plt.savefig(fft3_path, bbox_inches='tight')
            plt.close()

            # 4) Spectru perturbat + prag α
            plt.figure(figsize=(8, 3))
            plt.plot(np.abs(fs), label='|FFT(s)| – perturbat')
            plt.plot([alfa] * N, 'r--', label=f'Prag α={alfa}')
            plt.title('4) Spectru perturbat și prag α')
            plt.legend()
            plt.grid(True)
            fft4_path = os.path.join(plots_dir, 'l2_fft4.png')
            plt.savefig(fft4_path, bbox_inches='tight')
            plt.close()

            # 5) Spectru semnal filtrat
            plt.figure(figsize=(8, 3))
            plt.plot(np.abs(H), 'g')
            plt.title('5) Spectru semnal filtrat')
            plt.grid(True)
            fft5_path = os.path.join(plots_dir, 'l2_fft5.png')
            plt.savefig(fft5_path, bbox_inches='tight')
            plt.close()

            # 6) Semnal util vs filtrat
            plt.figure(figsize=(8, 3))
            plt.plot(np.real(h), 'k', label='Semnal filtrat')
            plt.plot(q, ':g', label='Semnal util (referință)')
            plt.title('6) Semnal util vs filtrat')
            plt.legend()
            plt.grid(True)
            fft6_path = os.path.join(plots_dir, 'l2_fft6.png')
            plt.savefig(fft6_path, bbox_inches='tight')
            plt.close()

            # URL-uri către imagini
            plot_x_url = url_for('static', filename='plots/l2/l2_x.png')
            plot_lp_url = url_for('static', filename='plots/l2/l2_lp.png')
            plot_hp_url = url_for('static', filename='plots/l2/l2_hp.png')
            plot_cornell_url = url_for('static', filename='plots/l2/l2_cornell.png')

            plot_fft1_url = url_for('static', filename='plots/l2/l2_fft1.png')
            plot_fft2_url = url_for('static', filename='plots/l2/l2_fft2.png')
            plot_fft3_url = url_for('static', filename='plots/l2/l2_fft3.png')
            plot_fft4_url = url_for('static', filename='plots/l2/l2_fft4.png')
            plot_fft5_url = url_for('static', filename='plots/l2/l2_fft5.png')
            plot_fft6_url = url_for('static', filename='plots/l2/l2_fft6.png')

    return render_template(
        'l2.html',
        methods=METHODS,
        active_method=2,
        current_file_name=current_file_name,
        n_rows=n_rows,
        n_cols=n_cols,
        plot_x_url=plot_x_url,
        plot_lp_url=plot_lp_url,
        plot_hp_url=plot_hp_url,
        plot_cornell_url=plot_cornell_url,
        plot_fft1_url=plot_fft1_url,
        plot_fft2_url=plot_fft2_url,
        plot_fft3_url=plot_fft3_url,
        plot_fft4_url=plot_fft4_url,
        plot_fft5_url=plot_fft5_url,
        plot_fft6_url=plot_fft6_url,
    )


