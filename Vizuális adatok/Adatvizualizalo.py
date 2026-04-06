import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def statisztikageneralo(fajlnev='faradtsagnaplo.json'):
    try:
        with open(fajlnev, 'r', encoding='utf-8') as f:
            adatok= json.load(f)

        if not adatok:
            print("A napló üres!")
            return

        tablazat = pd.DataFrame(adatok)
        tablazat['DateTime'] = pd.to_datetime(tablazat['DateTime'])

        plt.figure(figsize = (12,8))

        plt.plot(tablazat['DateTime'], tablazat['EAR'], label ='EAR', color = 'blue', alpha=0.6, linewidth=2)

        for i, row in tablazat.iterrows():
            if row['Status'] == "Microsleep":
                plt.axvline(x=row['DateTime'], color='orange', linestyle='--', alpha=0.7,
                            label='Microsleep' if 'Microsleep' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif row['Status'] == "Yawn":
                plt.scatter(row['DateTime'], row['EAR'], color='orange', s=100,
                        label='Yawn' if 'Yawn' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif row['Status'] == "High BPM!":
                plt.scatter(row['DateTime'], row['EAR'], color='orange', marker='x',
                        label='High BPM' if 'High BPM' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif row['Status'] == "Sleep":
                plt.scatter(row['DateTime'], row['EAR'], color='red', linestyle='--', alpha=1,
                        label='Sleep' if 'Sleep' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title('Éberségi Statisztika az Idő Függvényében', fontsize=14)
        plt.xlabel('Time')
        plt.ylabel('EAR')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('"%Y-%m-%d %H:%M:%S"'))
        plt.gcf().autofmt_xdate()

        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()

        plt.savefig('statisztika.png')
        print("A statisztika elmentve: statisztika.png")
        plt.show()

    except FileNotFoundError:
        print(f"Hiba: A '{fajlnev}' fájl nem található. Indítsd el előbb a főprogramot!")

    except Exception as e:
        print(f"Hiba történt: {e}")

if __name__ == "__main__":
        statisztikageneralo()