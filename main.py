import mujoco
import mujoco.viewer
import time
import os

# Il file XML della scena che include il robot e l'ambiente.
xml_file = './models/terrain_1_scene.xml'
# Ottieni il percorso assoluto del file XML.
# __file__ è il percorso dello script corrente (main.py).
# os.path.dirname(__file__) è la directory 'src'.
# os.path.join(...) costruisce il percorso corretto per robot.mujoco.xml.
xml_path = os.path.join(os.path.dirname(__file__), xml_file)

# Carica il modello MuJoCo dal file XML.
try:
    model = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    exit()

# Crea un'istanza dei dati di simulazione.
data = mujoco.MjData(model)

# Lancia il visualizzatore passivo.
print("Avvio del visualizzatore MuJoCo. Chiudere la finestra per terminare.")
with mujoco.viewer.launch_passive(model, data) as viewer:
  # Ciclo di simulazione principale.
  while viewer.is_running():
    step_start = time.time()

    # Esegui un passo della simulazione.
    mujoco.mj_step(model, data)

    # Sincronizza il visualizzatore con i dati di simulazione.
    viewer.sync()

    # Attendi per mantenere la simulazione circa in tempo reale.
    # model.opt.timestep è l'intervallo di tempo della simulazione (definito in scene.xml).
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

print("Visualizzatore chiuso. Programma terminato.")