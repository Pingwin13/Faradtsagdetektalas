import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def generate_statistics(filename='fatigue_log.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            print("The log file is empty!")
            return

        # JSON-ből Pandas DataFrame-be
        table = pd.DataFrame(data)

        # Kötelező oszlopok ellenőrzése és hibakezelés
        required_columns = {'DateTime', 'Status', 'EAR'}
        if not required_columns.issubset(table.columns):
            print("Error: The JSON file does not contain the required fields (DateTime, Status, EAR).")
            return

        # Dátum konvertálás és rendezés
        table['DateTime'] = pd.to_datetime(table['DateTime'])
        table = table.sort_values('DateTime')

        # Ábra létrehozása
        plt.figure(figsize=(14, 8))

        # EAR görbe
        plt.plot(
            table['DateTime'],
            table['EAR'],
            label='EAR',
            color='blue',
            alpha=0.75,
            linewidth=2
        )

        # Már kirajzolt jelmagyarázat elemek nyilvántartása
        plotted_labels = set()

        # Események megjelenítése
        for _, row in table.iterrows():
            status = row['Status']
            dt = row['DateTime']
            ear = row['EAR']

            if status == "Microsleep":
                label = "Microsleep" if "Microsleep" not in plotted_labels else None
                plt.axvline(x=dt, color='orange', linestyle='--', alpha=0.8, label=label)
                plotted_labels.add("Microsleep")

            elif status == "Yawn":
                label = "Yawn" if "Yawn" not in plotted_labels else None
                plt.scatter(dt, ear, color='orange', s=100, marker='o', label=label, zorder=5)
                plotted_labels.add("Yawn")

            elif status == "High BPM":
                label = "High BPM" if "High BPM" not in plotted_labels else None
                plt.scatter(dt, ear, color='purple', s=100, marker='x', label=label, zorder=5)
                plotted_labels.add("High BPM")

            elif status == "Sleep":
                label = "Sleep" if "Sleep" not in plotted_labels else None
                plt.axvline(x=dt, color='red', linestyle='-', alpha=0.9, label=label)
                plotted_labels.add("Sleep")

            elif status == "Awake":
                label = "Awake" if "Awake" not in plotted_labels else None
                plt.axvline(x=dt, color='green', linestyle='-', alpha=0.9, label=label)
                plotted_labels.add("Awake")

            elif status == "Head tilt":
                label = "Head tilt" if "Head tilt" not in plotted_labels else None
                plt.scatter(dt, ear, color='red', s=50, marker='.', label=label, alpha=0.7)
                plotted_labels.add("Head tilt")

            elif status == "Blink":
                label = "Blink" if "Blink" not in plotted_labels else None
                plt.scatter(dt, ear, color='green', s=50, marker='x', label=label, alpha=0.7)
                plotted_labels.add("Blink")

        # Címek és tengelyek
        plt.title('Alertness Statistics Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('EAR Value', fontsize=12)

        # Dátumformázás
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.gcf().autofmt_xdate()

        # Rács, legenda, margók
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # Statisztikai összegzés
        print("\n--- Summary ---")
        print(f"Total number of events: {len(table)}")
        print(f"Average EAR: {table['EAR'].mean():.3f}")
        print(f"Minimum EAR: {table['EAR'].min():.3f}")
        print(f"Maximum EAR: {table['EAR'].max():.3f}")

        # Mentés
        plt.savefig('statistics.png', dpi=300)
        print("The statistics have been saved as: statistics.png")

        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found. Please run the main program first!")

    except json.JSONDecodeError:
        print(f"Error: The file '{filename}' is corrupted or has an invalid JSON format.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_statistics()