import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

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

        plt.plot(tablazat['DateTime'], tablazat['Eye'], label ='EAR (Szemnyitottság)', color = 'blue', alpha=0.6, linewidth=2)

        for i, row in tablazat.iterrows():
            if row['Status'] == "MICROSLEEP!":
                plt.axvline(x=row['DateTime'], color='red', linestyle='--', alpha=0.7,
                            label='Microsleep' if 'Microsleep' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif row['Status'] == "Asitas":
                plt.scatter(row['DateTime'], row['Eye'], color='orange', s=100,
                        label='Ásítás' if 'Ásítás' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif row['Status'] == "Szemfaradtsag (Magas BPM)":
                plt.scatter(row['DateTime'], row['Eye'], color='purple', marker='x',
                        label='Magas BPM' if 'Magas BPM' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title('Éberségi Statisztika az Idő Függvényében', fontsize=14)
        plt.xlabel('Időpont')
        plt.ylabel('EAR Érték')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
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