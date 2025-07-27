import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import imageio.v2 as imageio # Usamos v2 para compatibilidad
import os # Para manejar archivos y directorios
import shutil # Para eliminar directorios completos

# --- Parámetros Globales del Sistema ---
NOMINAL_FREQUENCY_HZ = 50.0 # Frecuencia nominal del sistema en Hz
NOMINAL_OMEGA = 2 * np.pi * NOMINAL_FREQUENCY_HZ # Velocidad angular nominal en rad/s
S_BASE_MVA = 100.0 # Potencia base en MVA

# Parámetros de la Máquina Síncrona (SM)
H_SM = 5.0 # Constante de inercia en segundos (típica: 2-10 s)
D_SM = 0.09 # Coeficiente de amortiguamiento (FIJO A 0.09)
X_d_prime_SM = 0.2 # Reactancia transitoria del eje directo (p.u.)
E_q_prime_SM = 1.1 # Voltaje transitorio interno (EMF detrás de Xd') (p.u.)

# Parámetros del Bus Infinito (Grid)
V_BUS = 1.0 # Magnitud de voltaje del bus infinito (p.u.)
DELTA_BUS = 0.0 # Ángulo del bus infinito (rad)

# Parámetros de la Línea de Transmisión (entre SM y Bus)
X_LINE = 0.05 # Reactancia de la línea (p.u.)

# --- Configuración para GIF ---
GIF_FPS = 10 # Frames por segundo del GIF (ajustado para menor duración total)
# ¡RUTA DE SALIDA DEFINIDA AQUÍ!
OUTPUT_BASE_DIR = "output_gifs" # Directorio para guardar los GIFs
FRAMES_SUBDIR_NAME = "sm_simulation_frames_temp" # Subdirectorio temporal para frames individuales

# --- Función para guardar un frame de la gráfica ---
def plot_and_save_frame(time_data, delta_data, freq_data, Pe_data, Pm_data,
                        current_time, title, frame_num, total_duration, output_path,
                        H_val, D_val, Xd_prime_val, Eq_prime_val,
                        current_Pm_display, current_X_line_display):

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Asegúrate de que los datos de tiempo no sean solo un punto
    if len(time_data) < 2:
        # Si solo tenemos un punto, extendemos para que plot no falle
        time_data = np.append(time_data, time_data + 1e-6)
        delta_data = np.append(delta_data, delta_data)
        freq_data = np.append(freq_data, freq_data)
        Pe_data = np.append(Pe_data, Pe_data)
        Pm_data = np.append(Pm_data, Pm_data)

    # Gráfico del Ángulo del Rotor
    axes[0].plot(time_data, np.degrees(delta_data), 'b-', label='Ángulo del Rotor $\delta$ (grados)')
    axes[0].axvline(x=current_time, color='gray', linestyle='--', linewidth=1, label='Tiempo Actual')
    axes[0].set_ylabel('Ángulo (grados)')
    axes[0].grid(True)
    axes[0].legend(loc='upper right')
    axes[0].set_title(r'Ángulo del Rotor ($\delta$)')
    axes[0].set_xlim(0, total_duration) # Fija el eje X para toda la duración
    axes[0].autoscale_view()


    # Gráfico de Frecuencia
    axes[1].plot(time_data, freq_data, 'g-', label='Frecuencia (Hz)')
    axes[1].axvline(x=current_time, color='gray', linestyle='--', linewidth=1)
    axes[1].axhline(y=NOMINAL_FREQUENCY_HZ, color='r', linestyle='--', label='Frecuencia Nominal')
    axes[1].set_ylabel('Frecuencia (Hz)')
    axes[1].grid(True)
    axes[1].legend(loc='upper right')
    axes[1].set_title('Frecuencia del Sistema')
    axes[1].set_xlim(0, total_duration) # Fija el eje X para toda la duración
    axes[1].autoscale_view()


    # Gráfico de Potencias
    axes[2].plot(time_data, Pm_data, 'k--', label='Potencia Mecánica $P_m$ (p.u.)')
    axes[2].plot(time_data, Pe_data, 'm-', label='Potencia Eléctrica $P_e$ (p.u.)')
    axes[2].axvline(x=current_time, color='gray', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Tiempo (s)')
    axes[2].set_ylabel('Potencia (p.u.)')
    axes[2].grid(True)
    axes[2].legend(loc='upper right')
    axes[2].set_title('Balance de Potencias')
    axes[2].set_xlim(0, total_duration) # Fija el eje X para toda la duración
    axes[2].autoscale_view()

    # Título general de la figura incluyendo los parámetros específicos de esta simulación
    fig.suptitle(
        f"{title}\n Tiempo: {current_time:.2f} s\n"
        r"$H$={:.2f} s, $D$={:.2f}, $X_d'$={:.2f} p.u., $E_q'$={:.2f} p.u.".format(
            H_val, D_val, Xd_prime_val, Eq_prime_val) +
        f"\n$P_m$ actual: {current_Pm_display:.2f} p.u., $X_{{line}}$ actual: {current_X_line_display:.2f} p.u.",
        fontsize=16
    )

    # Añadir la Ecuación de Oscilación en formato LaTeX
    equation_text = r"$\frac{2H}{\omega_0} \frac{d^2\delta}{dt^2} = P_m - P_e - D\frac{d\delta}{dt}$"
    fig.text(0.5, 0.04, equation_text, horizontalalignment='center',
             verticalalignment='bottom', transform=fig.transFigure, fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.7))

    # Descripciones de variables (ajustadas para el espacio en el pie de página)
    variable_desc_text = (
        r"$\mathbf{H}$: Inercia (s) | "
        r"$\mathbf{\omega_0}$: Vel. angular nominal (rad/s) | "
        r"$\mathbf{\delta}$: Ángulo rotor (rad) "
        r"$\mathbf{P_m}$: Pot. mecánica (p.u.) | "
        r"$\mathbf{P_e}$: Pot. eléctrica (p.u.) | "
        r"$\mathbf{D}$: Amortiguamiento"
    )
    fig.text(0.5, 0.005, variable_desc_text, horizontalalignment='center',
             verticalalignment='bottom', transform=fig.transFigure, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.2", fc="lightgray", ec="black", lw=0.5, alpha=0.8))

    # Display de valores instantáneos de las variables de estudio
    # Asegúrate de que time_data tenga al menos un punto
    if len(time_data) > 0:
        current_delta_deg = np.degrees(delta_data[-1])
        current_freq_hz = freq_data[-1]
        current_Pe_val = Pe_data[-1]
        current_Pm_val = Pm_data[-1] # Pm ya se está mostrando en el título, pero lo incluimos para consistencia.

        values_display_text = (
            f"$\delta$: {current_delta_deg:.2f}°\n"
            f"$f$: {current_freq_hz:.2f} Hz\n"
            f"$P_m$: {current_Pm_val:.2f} p.u.\n"
            f"$P_e$: {current_Pe_val:.2f} p.u."
        )
        # Posición para los valores instantáneos (arriba a la derecha)
        fig.text(0.98, 0.96, values_display_text, horizontalalignment='right',
                 verticalalignment='top', transform=fig.transFigure, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=0.8, alpha=0.8))

        # Display de valores máximos y mínimos acumulados
        max_delta_deg = np.degrees(np.max(delta_data))
        min_delta_deg = np.degrees(np.min(delta_data))
        max_freq_hz = np.max(freq_data)
        min_freq_hz = np.min(freq_data)
        max_Pe_val = np.max(Pe_data)
        min_Pe_val = np.min(Pe_data)
        max_Pm_val = np.max(Pm_data)
        min_Pm_val = np.min(Pm_data)

        min_max_display_text = (
            f"Min/Max $\delta$: {min_delta_deg:.2f}° / {max_delta_deg:.2f}°\n"
            f"Min/Max $f$: {min_freq_hz:.2f} Hz / {max_freq_hz:.2f} Hz\n"
            f"Min/Max $P_m$: {min_Pm_val:.2f} p.u. / {max_Pm_val:.2f} p.u.\n"
            f"Min/Max $P_e$: {min_Pe_val:.2f} p.u. / {max_Pe_val:.2f} p.u."
        )
        # Posición para los valores min/max (arriba a la izquierda, debajo del título)
        fig.text(0.02, 0.96, min_max_display_text, horizontalalignment='left',
                 verticalalignment='top', transform=fig.transFigure, fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", lw=0.8, alpha=0.8))


    plt.tight_layout(rect=[0, 0.07, 1, 0.94]) # Ajuste para dar espacio a las descripciones

    # Crea el directorio si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Guarda el frame
    frame_filename = os.path.join(output_path, f"frame_{frame_num:04d}.png")
    plt.savefig(frame_filename)
    plt.close(fig) # Cierra la figura para liberar memoria


# --- Función de Ecuaciones Diferenciales de la SM (sin cambios) ---

def synchronous_machine_odes(t, Y, P_m, P_e_func, H, D, omega0_rad_s):
    """
    Define el sistema de ecuaciones diferenciales para la máquina síncrona.

    Y = [delta, omega]
    """
    delta, omega = Y[0], Y[1]

    # Derivada del ángulo del rotor
    d_delta_dt = omega - omega0_rad_s

    # Potencia eléctrica (depende del ángulo actual y otros parámetros)
    P_e = P_e_func(delta) # Se pasa como función para que pueda acceder a los estados del bus/línea

    # Derivada de la velocidad angular del rotor (Ecuación de Oscilación)
    # 2H/omega0 * d(omega)/dt = Pm - Pe - D(omega - omega0)
    # d(omega)/dt = (omega0 / 2H) * (Pm - Pe - D(omega - omega0))
    d_omega_dt = (omega0_rad_s / (2 * H)) * (P_m - P_e - D * (omega - omega0_rad_s))

    return [d_delta_dt, d_omega_dt]

# --- Función para Simular el Sistema y Generar Frames ---

def simulate_sm_and_get_frames(
    scenario_name_base, # Nombre base para el escenario
    sim_index, # Índice de la simulación (para el nombre del archivo)
    duration_s,
    initial_P_m,
    mech_power_changes,
    H,
    D,
    Xd_prime,
    Eq_prime,
    X_line,
    line_reactance_changes=None
):

    full_scenario_name = f"{scenario_name_base}"
    print(f"\n--- Simulación: {full_scenario_name} ---")

    # Condición inicial: sistema en equilibrio (frecuencia nominal, P_e = P_m)
    total_reactance_initial = Xd_prime + X_line
    # Asegúrate de que el argumento de arcsin esté en el rango [-1, 1]
    sin_delta_arg = initial_P_m * total_reactance_initial / (Eq_prime * V_BUS)
    if abs(sin_delta_arg) > 1.0:
        # Ajusta ligeramente si está fuera de rango debido a pequeños errores de punto flotante
        sin_delta_arg = np.sign(sin_delta_arg) * 1.0
    initial_delta = np.arcsin(sin_delta_arg) + DELTA_BUS

    Y0 = [initial_delta, NOMINAL_OMEGA]

    # Listas para almacenar los resultados completos (para el plot final y frames)
    full_time_points = []
    full_delta_history = []
    full_omega_history = []
    full_Pe_history = []
    full_Pm_history = []

    # Rutas para los frames temporales de este escenario
    # Se reemplazan caracteres especiales para asegurar que sea un nombre de directorio válido
    current_frames_dir = os.path.join(OUTPUT_BASE_DIR, FRAMES_SUBDIR_NAME, full_scenario_name.replace(' ', '_').replace(':', '').replace('.', '').replace('(','').replace(')',''))


    frame_files = []
    frame_num = 0
    capture_interval = 1.0 / GIF_FPS # Cuánto tiempo simular por frame

    current_P_m = initial_P_m
    current_X_line = X_line
    current_time = 0.0
    current_Y = np.array(Y0)

    all_events = []
    all_events.extend([(t, 'mech', val) for t, val in mech_power_changes])
    if line_reactance_changes:
        all_events.extend([(t, 'line', val) for t, val in line_reactance_changes])
    all_events.sort(key=lambda x: x[0])

    all_events.append((duration_s + 1e-9, 'end', None)) # Añade un evento final para asegurar que se procesa todo

    event_index = 0

    while current_time < duration_s:
        next_event_time = all_events[event_index][0] if event_index < len(all_events) else duration_s + 1e-9

        # Determina el próximo punto de parada para la integración
        # Esto puede ser el próximo evento, o el próximo intervalo de captura de frame, o el final de la simulación
        t_target_integration = min(duration_s, next_event_time, current_time + capture_interval)

        # Si el tiempo objetivo es el mismo que el tiempo actual, avanzamos al siguiente evento/paso
        if t_target_integration <= current_time:
            # Si estamos en un tiempo de evento, aplicarlo y continuar
            if event_index < len(all_events) and abs(current_time - all_events[event_index][0]) < 1e-6:
                event_time_actual, event_type, event_value = all_events[event_index]
                if event_type == 'mech':
                    current_P_m = event_value
                elif event_type == 'line':
                    current_X_line = event_value
                event_index += 1
            current_time = t_target_integration
            continue


        t_span = [current_time, t_target_integration]

        def P_e_at_delta(delta_val):
            total_reactance = Xd_prime + current_X_line
            if abs(total_reactance) < 1e-6: return 0.0
            return (Eq_prime * V_BUS / total_reactance) * np.sin(delta_val - DELTA_BUS)

        sol = solve_ivp(
            fun=lambda t_ode, Y_ode: synchronous_machine_odes(t_ode, Y_ode, current_P_m, P_e_at_delta, H, D, NOMINAL_OMEGA),
            t_span=t_span,
            y0=current_Y,
            method='RK45',
            dense_output=True,
            max_step=0.005 # Paso más pequeño para mejor resolución en GIF
        )

        if not sol.success:
            print(f"ERROR: La simulación falló en t={current_time:.2f}s - {sol.message}")
            break

        # Añadir resultados al historial, evitando duplicados en los puntos de inicio/fin de segmento
        # Buscar el punto de inicio del segmento actual en el historial completo
        start_idx_in_sol_t = 0
        if full_time_points and sol.t[0] == full_time_points[-1]:
            start_idx_in_sol_t = 1 # Saltar el primer punto si es duplicado

        full_time_points.extend(sol.t[start_idx_in_sol_t:])
        full_delta_history.extend(sol.y[0, start_idx_in_sol_t:])
        full_omega_history.extend(sol.y[1, start_idx_in_sol_t:])
        full_Pe_history.extend([P_e_at_delta(d) for d in sol.y[0, start_idx_in_sol_t:]])
        full_Pm_history.extend([current_P_m] * (len(sol.t) - start_idx_in_sol_t))

        current_time = sol.t[-1]
        current_Y = sol.y[:, -1]

        # Guardar frame para el GIF en el directorio temporal
        plot_and_save_frame(
            np.array(full_time_points),
            np.array(full_delta_history),
            np.array(full_omega_history) / (2 * np.pi),
            np.array(full_Pe_history),
            np.array(full_Pm_history),
            current_time,
            full_scenario_name,
            frame_num,
            duration_s,
            current_frames_dir, # Pasa la ruta del subdirectorio temporal
            H, D, Xd_prime, Eq_prime, # Pasa los parámetros para la etiqueta
            current_P_m, current_X_line # Pasa los valores actuales de Pm y X_line
        )
        frame_files.append(os.path.join(current_frames_dir, f"frame_{frame_num:04d}.png"))
        frame_num += 1

        # Si el próximo evento está cerca del current_time, lo aplicamos
        if event_index < len(all_events) and abs(current_time - all_events[event_index][0]) < 1e-6:
            event_time_actual, event_type, event_value = all_events[event_index]
            if event_type == 'mech':
                current_P_m = event_value
            elif event_type == 'line':
                current_X_line = event_value
            event_index += 1
    
    return frame_files, current_frames_dir


# --- Escenarios de Simulación ---

if __name__ == "__main__":
    # Crea el directorio principal de salida si no existe
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)

    all_frames_for_combined_gif = []
    temp_dirs_to_clean = []

    # Duración de cada escenario REDUCIDA a 10 segundos
    scenario_duration = 10.0 # segundos por escenario
    perturbation_time = 5.0 # tiempo de la perturbación en cada escenario (mitad de la duración)

    # Escenario 1: Comportamiento Base (usando parámetros nominales)
    initial_P_mech_base = 0.8
    mech_power_changes_base = [
        (perturbation_time, 0.6),
        (perturbation_time + 0.1, 0.8) # Recuperación rápida
    ]
    frames, temp_dir = simulate_sm_and_get_frames(
        "1. Comportamiento Base (Pm a 0.6)", 1,
        duration_s=scenario_duration,
        initial_P_m=initial_P_mech_base,
        mech_power_changes=mech_power_changes_base,
        H=H_SM, D=D_SM,
        Xd_prime=X_d_prime_SM, Eq_prime=E_q_prime_SM,
        X_line=X_LINE
    )
    all_frames_for_combined_gif.extend(frames)
    temp_dirs_to_clean.append(temp_dir)

    # Escenario 2: Mayor Inercia (H) - usando H_scenario_2_base
    H_scenario_2_base = 8.0 # Mayor inercia
    frames, temp_dir = simulate_sm_and_get_frames(
        "2. Mayor Inercia (H=8s, Pm a 0.6)", 1,
        duration_s=scenario_duration,
        initial_P_m=initial_P_mech_base,
        mech_power_changes=mech_power_changes_base,
        H=H_scenario_2_base, D=D_SM,
        Xd_prime=X_d_prime_SM, Eq_prime=E_q_prime_SM,
        X_line=X_LINE
    )
    all_frames_for_combined_gif.extend(frames)
    temp_dirs_to_clean.append(temp_dir)

    # Escenario 3: Amortiguamiento Reducido (D) - simulando un D más bajo para mostrar el efecto
    D_scenario_3_low = 0.02
    mech_power_changes_D_low = [
        (perturbation_time, 0.6),
        (perturbation_time + 0.1, 0.8)
    ]
    frames, temp_dir = simulate_sm_and_get_frames(
        "3. Menor Amortiguamiento (D=0.02, Pm a 0.6)", 1,
        duration_s=scenario_duration,
        initial_P_m=initial_P_mech_base,
        mech_power_changes=mech_power_changes_D_low,
        H=H_SM, D=D_scenario_3_low, # Aquí usamos un D menor
        Xd_prime=X_d_prime_SM, Eq_prime=E_q_prime_SM,
        X_line=X_LINE
    )
    all_frames_for_combined_gif.extend(frames)
    temp_dirs_to_clean.append(temp_dir)

    # Escenario 4: Reactancia Transitoria Reducida (Xd')
    Xd_prime_scenario_4_base = 0.1 # Menor reactancia transitoria
    frames, temp_dir = simulate_sm_and_get_frames(
        "4. Reactancia Transitoria Reducida (Xd'=0.1, Pm a 0.6)", 1,
        duration_s=scenario_duration,
        initial_P_m=initial_P_mech_base,
        mech_power_changes=mech_power_changes_base,
        H=H_SM, D=D_SM,
        Xd_prime=Xd_prime_scenario_4_base, Eq_prime=E_q_prime_SM,
        X_line=X_LINE
    )
    all_frames_for_combined_gif.extend(frames)
    temp_dirs_to_clean.append(temp_dir)

    # Escenario 5: Inestabilidad por Alta Demanda
    initial_P_mech_high = 0.9
    mech_power_changes_high_load = [
        (perturbation_time, 1.2), # Aumento significativo de demanda que puede causar inestabilidad
        (perturbation_time + 0.1, 0.9) # Intento de recuperación, pero podría no ser suficiente
    ]
    frames, temp_dir = simulate_sm_and_get_frames(
        "5. Inestabilidad por Alta Demanda (Pm a 1.2)", 1,
        duration_s=scenario_duration,
        initial_P_m=initial_P_mech_high,
        mech_power_changes=mech_power_changes_high_load,
        H=H_SM, D=D_SM,
        Xd_prime=X_d_prime_SM, Eq_prime=E_q_prime_SM,
        X_line=X_LINE
    )
    all_frames_for_combined_gif.extend(frames)
    temp_dirs_to_clean.append(temp_dir)

    # Escenario 6: Cortocircuito Cercano (Falla y Recuperación)
    initial_P_mech_fault = 0.8
    line_changes_for_fault = [
        (perturbation_time, 100.0), # Simula una reactancia muy alta (cortocircuito)
        (perturbation_time + 0.5, 0.05) # Recuperación de la línea después de 0.5s (despeje de falla)
    ]
    frames, temp_dir = simulate_sm_and_get_frames(
        "6. Cortocircuito y Recuperación (Xl a 100 p.u.)", 1,
        duration_s=scenario_duration,
        initial_P_m=initial_P_mech_fault,
        mech_power_changes=[],
        line_reactance_changes=line_changes_for_fault,
        H=H_SM, D=D_SM,
        Xd_prime=X_d_prime_SM, Eq_prime=E_q_prime_SM,
        X_line=X_LINE
    )
    all_frames_for_combined_gif.extend(frames)
    temp_dirs_to_clean.append(temp_dir)

    # --- Crear un único GIF combinado ---
    if all_frames_for_combined_gif:
        combined_gif_filename = os.path.join(OUTPUT_BASE_DIR, "Todos_los_Escenarios_Estabilidad_SM.gif")
        print(f"\n--- Creando GIF combinado: {combined_gif_filename} ---")
        try:
            with imageio.get_writer(combined_gif_filename, mode='I', fps=GIF_FPS) as writer:
                for filename in all_frames_for_combined_gif:
                    try:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                    except Exception as e:
                        print(f"Error al leer la imagen {filename}: {e}")
            print(f"GIF combinado creado exitosamente en: {combined_gif_filename}")
        except Exception as e:
            print(f"ERROR: No se pudo crear el GIF combinado {combined_gif_filename}: {e}")
        finally:
            # Limpiar todos los directorios temporales
            for d in temp_dirs_to_clean:
                if os.path.exists(d):
                    shutil.rmtree(d)
                    print(f"Directorio temporal '{d}' eliminado.")
    else:
        print("No se generaron frames para el GIF combinado.")
