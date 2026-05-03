# Fáradtságdetektáló rendszer prototípus

Ez a projekt egy valós idejű, számítógépes látáson alapuló proof of concept fáradtságdetektáló rendszer. 
A script a kamera képét elemezve, a MediaPipe Face Mesh és az OpenCV segítségével figyeli a felhasználó arcát,
és hangriasztást ad, ha mikroalvás vagy elalvás jeleit észleli. 
Ezenkívül percenkénti pislogást számlál és ásítást és fejdőlést is detektál.

# Faradtsagdetektalo.py
## Fő funkciók:
- Dinamikus kalibráció (első 100 frame, átlagos szemnyitottság rögzítése, fejtartás változásához szükséges adatok rögzítése, arcközéppont rögzítése)
- Pislogás (saját kalibrált nyitott szemérték 65%-a alatt) és szemcsukódás figyelés (Alvás állapot 15 mp. után mikroalvás 3 mp. után)
- Ásítás detektálás (magasabb, mint 0.5-ös MAR érték legalább 60 Frame-en keresztül)
- Fejdőlés vizsgálat (kalibrációkor rögzített szögekhez képest)
- Hangriasztás (Microsleep és Sleep állapotoknál)
- Eseménynaplózás (állapotváltozások rögzítése JSON fájlba, Statusz - időbélyeg - EAR érték) 


## Rendszerkövetelmény:
### Operációs rendszer: 
- Windows
### Hardver:
- Működő webkamera
### Python:
- Python 3.8 vagy újabb verzió

## Szükséges Python könyvtárak:
pip install opencv-python mediapipe numpy
( A math, time, collections, threading, json, datetime, os, winsound standard könyvtárak részei)

## Használat:
### 1. Kamera csatlakoztatása
### 2. A script futtatása
### 3. Kalibrációs fázis: 
- Nézz egyenesen a kamerába, amíg a kalibráció be nem fejeződik
### 4. Működés: 
- A kalibráció befejeztével a rendszer elkezdi a valós idejű elemzést, a képernyőn láthatod a különböző adatokat.

### Gombok:
- C - újrakalibrálás

- ESC - kilépés

## Naplózás:
A rendszer automatikusan létrehoz egy Vizuális adatok nevű mappát a script futtatási helyén.
Ebbe a mappába generál egy fatigue_log.json nevű fájlt, amely időbélyeggel rögzíti a státuszváltozásokat és az abban a pillanatban mért EAR értéket.


# Adatvizualizalo.py

## Fő funkciók:
- EAR görbe kirajzolása
- Események vizualizációja
- Interaktív grafikon megjelenítés
- Szöveges összegzés a konzolon
- Automatikus mentés

## Szükséges csomagok:
pip install pandas matplotlib

Miután a Faradtsagdetektalo.py-t lefuttattad és különböző státuszok rögzültek a JSON fájlba, indítsd el az Adatvizualizalo.py-t.
(Győződj meg róla, hogy abban a mappában fut, ahonnan eléri a fatigue_log.json fájlt!)


# Fontos megjegyzések:
- ### A hangriasztás csak windowson működik
- ### A rendszer ideális fényviszonyok és frontális vagy kissé elfordított arcra van optimalizálva és tesztelve
- ### Több kamera esetén a cap = cv2.VideoCapture(0) sorba az értéket módosítsd 1-re vagy 2-re

