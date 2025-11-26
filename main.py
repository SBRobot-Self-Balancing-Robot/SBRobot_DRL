import mujoco
import mujoco.viewer
import time
import os
import numpy as np

# Il file XML della scena che include il robot e l'ambiente.
xml_file = './models/scene.xml'
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
pitch = 0.0
with mujoco.viewer.launch_passive(model, data) as viewer:
  # Ciclo di simulazione principale.
  while viewer.is_running():
    step_start = time.time()

    # Esegui un passo della simulazione.
    #data  # Esempio di azione: coppie di velocità per le ruote destra e sinistra.
    mujoco.mj_step(model, data)

    # Get the wheel positions
    accell = data.sensordata[model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer") : model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")] + 3]]
    gyro = data.sensordata[model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyroscope") : model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyroscope")] + 3]]
    pitch = 0.996 * (pitch + gyro[1] * model.opt.timestep) - 0.004 * np.arctan2(accell[0], accell[2])
    print(f"angolo: {pitch:.2f} rad")
    # Sincronizza il visualizzatore con i dati di simulazione.
    viewer.sync()

    # Attendi per mantenere la simulazione circa in tempo reale.
    # model.opt.timestep è l'intervallo di tempo della simulazione (definito in scene.xml).
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

print("Visualizzatore chiuso. Programma terminato.")