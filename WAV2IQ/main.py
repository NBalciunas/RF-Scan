import os
import numpy as np
import soundfile as sf
import xml.etree.ElementTree as ET

# CONFIG
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

CENTER_FREQUENCY = 2400000000  # Hz
SAMPLE_RATE_OVERRIDE = None
DEVICE_TYPE = "SDRSharp/PlutoSDR"
LABEL = "SIGNAL"
SERIAL_NUMBER = "00001"
REFERENCE_SNR = "40"
SCALE_FACTOR = "1.0"

# CONFIG END

def wav_to_iq_xml():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for file in os.listdir(INPUT_FOLDER):
        if not file.lower().endswith(".wav"):
            continue

        wav_path = os.path.join(INPUT_FOLDER, file)
        name = os.path.splitext(file)[0]

        data, sr = sf.read(wav_path, always_2d=True)

        if data.shape[1] != 2:
            raise ValueError(f"{file}: Expected stereo IQ WAV")

        iq = data.astype(np.float32)

        sample_rate = SAMPLE_RATE_OVERRIDE if SAMPLE_RATE_OVERRIDE else sr

        iq_filename = name + ".iq"
        iq.tofile(os.path.join(OUTPUT_FOLDER, iq_filename))

        root = ET.Element("SignalHoundIQFile")

        ET.SubElement(root, "DeviceType").text = DEVICE_TYPE
        ET.SubElement(root, "Drone").text = LABEL
        ET.SubElement(root, "SerialNumber").text = SERIAL_NUMBER
        ET.SubElement(root, "DataType").text = "Complex Float"
        ET.SubElement(root, "ReferenceSNRLevel").text = REFERENCE_SNR
        ET.SubElement(root, "CenterFrequency").text = str(CENTER_FREQUENCY)
        ET.SubElement(root, "SampleRate").text = str(sample_rate)
        ET.SubElement(root, "IFBandwidth").text = str(sample_rate)
        ET.SubElement(root, "ScaleFactor").text = SCALE_FACTOR
        ET.SubElement(root, "IQFileName").text = iq_filename
        ET.SubElement(root, "SampleCount").text = str(len(iq))

        ET.ElementTree(root).write(
            os.path.join(OUTPUT_FOLDER, name + ".xml")
        )

        print(f"Converted: {file}")


if __name__ == "__main__":
    wav_to_iq_xml()
