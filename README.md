# Faradsagdetektalas

Ez a Python nyelven írt alkalmazás egy MediaPipe Face Mesh-en alapuló fáradtságdetektáló alkalmazás, amely webkamerán keresztül valós időben detektálja a fáradtság jeleit.( ásítás, pislogás szám, szem állapota, fej pozíciója)

Funkcionalitás:
Indításkor a program egy kalibrációt végez a felhasználó egyedi szemfelépítéséről, így egyedi küszöbértékekkel működik. Abban az esetben, ha a kalibráció sikertelen, előre rögzített értékeket használ.
Valós idejű maszkot rajzol az arcra, és szöveges figyelmeztetéseket és értékeket jelenít meg.
Az Eye Aspect Ratio (EAR) algoritmus segítségével érzékeli a pislogást és a szem hosszabb ideig tartó lehunyását. 
A Mouth Aspect Ratio (MAR) mérésével figyeli a száj nyitottságát, és jelzi a gyakori ásítást.
Head Pose Estimation a 3D-s arcmodell alapján érzékeli, ha a felhasználó feje előrebukik vagy oldalra dől.
A EAR és a Head Pose Estimation közös értékein alapulva jelzi a képernyőre ( szem csukva van és a fej pozíciója nem normál állapotban van), hogy Alvás veszély áll fenn.
A pislogás statisztika (BPM) számolja a percenkénti pislogásszámot, amiből következtet a szem fáradtságára.

Felhasznált technológiák:
MediaPipe Face Mesh: 468+ pontos 3D-s archáló kinyerése.
OpenCV: Videófolyam kezelése és képmunkálás.
NumPy & Math: Matematikai számítások a 3D-s pontok távolságának és a fej dőlésszögének meghatározásához.
