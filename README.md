# Previsione dei Pit Stop in Formula 1 con un Multilayer Perceptron

Un progetto di machine learning che insegna a una rete neurale a capire quando un pilota di Formula 1 sta per fermarsi ai box. Il programma guarda la situazione di un pilota alla fine di ogni giro (che gomma ha, quanto è consumata, quanti giri mancano, se c'è la safety car) e risponde con una percentuale: quanto è probabile che il pilota rientri ai box al giro successivo. Per esempio, 90% vuol dire che quasi sicuramente il pilota si fermerà, mentre 2% vuol dire che quasi sicuramente resterà in pista. Se la percentuale è almeno del 50%, il programma risponde "rientra ai box"; altrimenti risponde "resta in pista".

Il progetto usa i dati reali di una gara, li trasforma in una tabella che un computer può leggere, addestra due modelli (un perceptron semplice e un Multilayer Perceptron), li confronta e spiega perché il secondo funziona meglio.

## Panoramica del progetto

### Il problema

In Formula 1 le gomme si consumano giro dopo giro e a un certo punto vanno cambiate. Il pilota rientra ai box, i meccanici montano gomme nuove e lui riparte, questo è il **pit stop**. La sosta costa più di 20 secondi, quindi scegliere il giro giusto può far vincere o perdere una gara. Fermarsi troppo presto spreca gomme ancora buone; fermarsi troppo tardi fa perdere secondi a ogni giro con gomme ormai finite.

A decidere è il **muretto box**, postazione in cui gli ingegneri della squadra guardano i dati durante la gara. La domanda di questo progetto è:

> **Una rete neurale può imparare da sola a riconoscere il momento del pit stop, guardando i dati di una gara vera?**

### Perché questa analisi

Il problema si presta bene a mostrare come funziona una rete neurale, per quattro motivi.

1. **È una domanda con due sole risposte**,al giro successivo il pilota si ferma (sì) oppure no? In machine learning si chiama **classificazione binaria**.
2. **I dati sono reali e pubblici**, quindi chiunque può scaricarli e rifare il progetto.
3. **I pit stop sono rari**: la gara ha 66 giri e in pista ci sono 22 piloti. Ogni giro di ogni pilota è un esempio per il modello, quindi in tutto sono più di mille esempi, per essere esatti 1164. Solo in 48 di questi il pilota si ferma ai box al giro successivo, cioè circa 4 su 100. Un modello che risponde sempre "resta in pista" avrebbe ragione il 96% delle volte, ma non prevederebbe mai un pit stop. Per questo bisogna addestrarlo e valutarlo con attenzione.
4. **I dati non si dividono con una semplice "linea retta"**: un modello semplice come il perceptron non basta, e serve una rete con più strati. Il progetto lo dimostra con un test apposito.

### Cosa fa il progetto, passo per passo

- **Scarica i dati**: in questo caso vengono utilizzati dei dati estrapolati dall'archivio gratuito [OpenF1](https://openf1.org)
- **Controlla i dati con la fonte ufficiale**: OpenF1 non è un servizio ufficiale, quindi le soste sono state confrontate con il riepilogo dei pit stop pubblicato dalla FIA a fine gara. In entrambi risultano 48 soste, quindi i dati su cui il modello impara sono corretti. I documenti della FIA servono solo per questo controllo e non entrano nel modello
- **Prepara il dataset**: trasforma i dati grezzi in una tabella con una riga per ogni pilota alla fine di ogni giro
- **Addestra un perceptron semplice**, il modello più elementare, usato come termine di paragone
- **Addestra un Multilayer Perceptron (MLP)**, la nostra rete neurale, scritta da zero
- **Confronta i due modelli** su piloti che non hanno mai visto
- **Verifica se i dati sono "linearmente separabili"**, cioè se basterebbe una retta per dividerli
- **Fa previsioni**: dato un pilota e un giro, dice se rientrerà ai box

La gara usata è il **Gran Premio di Barcellona-Catalunya del 14 giugno 2026**: 66 giri, vinto da Lewis Hamilton con una strategia a tre soste. È una gara con strategie molto diverse tra i piloti, in quella gara c'è stato chi si è fermato una volta, chi due, chi tre, quindi offre al modello esempi vari da cui imparare.

## Concetti di base

Qui trovi spiegate in modo semplice le parole che compaiono nel resto del documento.

- **Machine learning**: di solito, per far fare qualcosa a un computer, gli si scrivono le regole una per una, per esempio "se la gomma ha più di 20 giri, fermati". Con il machine learning si fa il contrario: gli si mostrano tanti casi già successi e il computer trova da solo le regole che li spiegano.
- **Classificazione**: vuol dire mettere ogni caso nella categoria giusta, come quando si separa la posta in "importante" e "da buttare". Nel nostro progetto le categorie sono due: **pit**, quando il pilota si ferma ai box al giro dopo, e **no pit**, quando resta in pista.
- **Feature**: sono le informazioni che il modello guarda prima di rispondere. È quello che guarderebbe un ingegnere al muretto box: che gomma monta il pilota, da quanti giri la usa, quanti giri mancano alla fine, se in pista c'è la Safety Car.
- **Etichetta**: è la risposta giusta di ogni caso, quella che sappiamo perché la gara è già finita. Vale **1** se il pilota si è fermato davvero al giro dopo e **0** se non si è fermato. Il modello impara proprio confrontando le sue risposte con le etichette.
- **Neurone artificiale**: è il mattoncino base di una rete neurale. Funziona come una bilancia: prende ogni informazione e le dà un **peso**, cioè decide quanto conta. Per esempio "gomma molto vecchia" può contare tanto, "gomma dura" poco. Poi somma tutto e tira fuori un numero tra 0 e 1, che indica quanto è probabile il pit stop.
- **Perceptron**: è il modello più semplice che esista, fatto da un neurone solo. Funziona bene quando la risposta dipende da una regola semplice, ma si blocca quando la situazione è più complicata, per esempio quando la stessa gomma vecchia a volte porta al pit e a volte no, a seconda di quanti giri mancano.
- **Multilayer Perceptron (MLP)**: è una rete fatta di più neuroni messi in fila su più livelli, chiamati **strati**. Il primo strato riceve le informazioni e l'ultimo dà la risposta. In mezzo c'è lo **strato nascosto**: si chiama così perché non si vede né all'inizio né alla fine, ma è lì che il modello combina le informazioni tra loro, per esempio "gomma vecchia **e** tanti giri ancora da fare". È questo che gli permette di capire situazioni che il perceptron da solo non riesce a gestire.
- **Addestramento**: è la fase in cui il modello impara. Guarda un caso, prova a rispondere, controlla l'etichetta e, se ha sbagliato, corregge un po' i pesi. Lo ripete migliaia di volte, e ogni volta sbaglia un po' meno.
- **Test**: è l'esame finale del modello. Gli si danno casi che non ha mai visto durante l'addestramento, nel nostro progetto i giri di piloti che aveva tenuto da parte, e si controlla quante risposte azzecca. Serve a capire se ha imparato davvero o se ha solo memorizzato gli esempi, come uno studente che sa a memoria gli esercizi del libro ma non sa risolverne uno nuovo.
- **Linearmente separabili**: immagina di disegnare su un foglio tutti i giri della gara, con un punto rosso per ogni giro in cui il pilota si ferma al giro dopo e un punto blu per tutti gli altri. Se riesci a tracciare una sola riga dritta che lascia tutti i rossi da una parte e tutti i blu dall'altra, i dati sono linearmente separabili e basta il perceptron. Nel nostro caso questa riga non esiste, perché i punti rossi e blu sono mescolati. Per questo serve l'MLP.

## Legenda dei piloti

Nei dati e nei risultati ogni pilota è indicato con una **sigla di tre lettere**, la stessa che si vede in TV durante le gare. Sono le prime tre lettere del cognome. Questa sigla è stata utilizzata per fare questa previsione, in modo da avere lo stesso tipo li linguaggio e sigle.

| Sigla | Pilota | Squadra | Numero |
|---|---|---|---|
| ALB | Alexander Albon | Williams | 23 |
| ALO | Fernando Alonso | Aston Martin | 14 |
| ANT | Kimi Antonelli | Mercedes | 12 |
| BEA | Oliver Bearman | Haas | 87 |
| BOR | Gabriel Bortoleto | Audi | 5 |
| BOT | Valtteri Bottas | Cadillac | 77 |
| COL | Franco Colapinto | Alpine | 43 |
| GAS | Pierre Gasly | Alpine | 10 |
| HAD | Isack Hadjar | Red Bull Racing | 6 |
| HAM | Lewis Hamilton | Ferrari | 44 |
| HUL | Nico Hülkenberg | Audi | 27 |
| LAW | Liam Lawson | Racing Bulls | 30 |
| LEC | Charles Leclerc | Ferrari | 16 |
| LIN | Arvid Lindblad | Racing Bulls | 41 |
| NOR | Lando Norris | McLaren | 1 |
| OCO | Esteban Ocon | Haas | 31 |
| PER | Sergio Pérez | Cadillac | 11 |
| PIA | Oscar Piastri | McLaren | 81 |
| RUS | George Russell | Mercedes | 63 |
| SAI | Carlos Sainz | Williams | 55 |
| STR | Lance Stroll | Aston Martin | 18 |
| VER | Max Verstappen | Red Bull Racing | 3 |

I 6 piloti usati per il test sono **HAM, ANT, LEC, VER, NOR e RUS**: il modello non vede i loro giri durante l'addestramento. Gli altri 16 servono per addestrarlo. Per cambiarli basta modificare la riga `TEST_DRIVERS` all'inizio di `train_mlp.py`, ma in quel caso cambiano anche i risultati.

## Struttura del progetto

```
f1-pitstop/
├── README.md                  # Questo file
├── LICENSE                    # Licenza MIT
├── requirements.txt           # Librerie Python necessarie
├── download_data.py           # 1. Scarica i dati della gara da OpenF1
├── build_dataset.py           # 2. Costruisce il dataset giro per giro
├── perceptron_classifier.py   # Il perceptron semplice (modello di confronto)
├── mlp_classifier.py          # La nostra rete neurale (MLP)
├── train_mlp.py               # 3. Addestra e valuta i due modelli
├── check_separability.py      # 4. Controlla se pit e no pit si separano con una retta
├── predict.py                 # 5. Previsione per un pilota a un certo giro
├── data/
│   ├── raw/                   # Dati grezzi scaricati (file JSON)
│   └── dataset.csv            # Dataset pronto per il modello
├── mlp_model.pkl              # Modello addestrato (generato)
├── scaler.pkl                 # Parametri di standardizzazione (generato)
└── loss_curve.png             # Grafico dell'apprendimento (generato)
```

Ogni file fa una cosa sola, e ognuno usa quello che ha prodotto il precedente:

1. `download_data.py` scarica i dati e li salva in `data/raw/`.
2. `build_dataset.py` li trasforma nella tabella `data/dataset.csv`.
3. `train_mlp.py` legge la tabella, addestra i modelli definiti in `perceptron_classifier.py` e `mlp_classifier.py`, li valuta e salva il modello migliore.
4. `check_separability.py` analizza la tabella per capire quanto è difficile il problema.
5. `predict.py` usa il modello salvato per rispondere su un pilota e un giro precisi.

## Caratteristiche

- **Dati reali e gratuiti**: provengono da OpenF1, che non richiede account; chiunque può riscaricarli
- **Dati verificati**: i 48 pit stop della gara coincidono con il documento ufficiale della FIA
- **Rete neurale scritta da zero**: nessuna libreria di deep learning, ogni passaggio è visibile nel codice
- **Gestione delle classi sbilanciate**: i pit stop sono rari (4% dei giri) e il modello è addestrato per non ignorarli
- **Valutazione onesta**: i piloti usati per il test non vengono mai visti durante l'addestramento
- **Confronto tra due modelli**: perceptron e MLP sono valutati sugli stessi dati
- **Analisi della separabilità**: un test dimostra perché un modello semplice non basta
- **Risultati riproducibili**: lo stesso seed dà sempre gli stessi numeri

## Installazione

### Prerequisiti

- Python 3.9 o superiore
- pip (il programma che installa le librerie Python)
- Consigliati: Anaconda e Visual Studio Code

### Preparare l'ambiente

1. In Anaconda Navigator, da *Environments → Create*, creare un ambiente chiamato `f1-pitstop`.
2. In *Home* selezionare l'ambiente `f1-pitstop` e lanciare Visual Studio Code.
3. In VS Code, da *File → Apri cartella*, scegliere la cartella `f1-pitstop`.
4. Aprire il terminale (*Terminale → Nuovo terminale*) e installare le librerie:

```
pip install -r requirements.txt
```

### Librerie necessarie

- **numpy**: calcoli con tabelle di numeri. È il "motore" della rete neurale
- **pandas**: gestione delle tabelle di dati, come un foglio Excel dentro Python
- **scikit-learn**: divisione dei dati, standardizzazione e calcolo delle metriche
- **matplotlib**: il grafico dell'apprendimento
- **requests**: download dei dati da internet
- **scipy**: il test di separabilità lineare

## Utilizzo

Dal terminale di VS Code, i comandi si lanciano in quest'ordine. Per lanciarli tutti insieme:

```
pip install -r requirements.txt && python build_dataset.py && python train_mlp.py && python check_separability.py && python predict.py --driver HAM --lap 10
```

Sul Mac, se `python` o `pip` non vengono trovati, si usano `python3` e `pip3`.

### 1. Scaricare i dati (facoltativo)

```
python download_data.py
```

I dati sono già nella cartella `data/raw`. Questo comando serve solo per riscaricarli da OpenF1.

### 2. Costruire il dataset

```
python build_dataset.py
```

Legge i dati grezzi e crea `data/dataset.csv`, la tabella su cui lavorano i modelli.

### 3. Addestrare e valutare i modelli

```
# Con le impostazioni di default
python train_mlp.py

# Con impostazioni personalizzate
python train_mlp.py --hidden_size 16 --learning_rate 0.05 --epochs 2000
```

**Parametri di addestramento**

- `--hidden_size`: numero di neuroni nello strato nascosto (default: 8)
- `--learning_rate`: quanto grandi sono le correzioni dei pesi a ogni passo (default: 0.1)
- `--epochs`: quante volte la rete ripassa tutti i dati (default: 1000)
- `--random_state`: seed per avere sempre gli stessi risultati (default: 42)

I 6 piloti usati per il test non si scelgono dal terminale: sono scritti all'inizio di `train_mlp.py`, nella riga `TEST_DRIVERS`.

Ogni volta il modello salvato viene sovrascritto. Per tornare a quello originale basta rilanciare `python train_mlp.py` senza parametri.

### 4. Controllare se i dati sono linearmente separabili

```
python check_separability.py
```

Prende tutti i giri, pit e no pit insieme, e verifica se una sola "retta" può dividere i due gruppi senza errori. Poi confronta quanti pit trovano il perceptron, la migliore retta possibile e l'MLP.

### 5. Fare una previsione

```
python predict.py --driver HAM --lap 10
```

**Parametri di previsione**

- `--driver`: sigla del pilota (es. `HAM` per Hamilton, `OCO` per Ocon, `VER` per Verstappen)
- `--lap`: il giro appena concluso; il modello prevede cosa succede nel giro successivo

## Formato dei dati

### Da dove vengono

I dati arrivano da [OpenF1](https://openf1.org/), un archivio gratuito con i dati di cronometraggio della Formula 1. Per la gara scarichiamo cinque tabelle:

| Tabella        | Cosa contiene                                                                                      | Righe | A cosa ci serve                                   |
|----------------|----------------------------------------------------------------------------------------------------|-------|---------------------------------------------------|
| `laps`         | il tempo di ogni giro di ogni pilota                                                               | 1.241 | capire quanto il pilota sta rallentando           |
| `stints`       | i periodi con lo stesso treno di gomme: mescola, primo e ultimo giro, età della gomma al montaggio | 70    | mescola, età della gomma, numero dello stint      |
| `pit`          | ogni ingresso ai box, con il giro in cui è avvenuto                                                | 48    | l'etichetta da indovinare                         |
| `race_control` | i messaggi della direzione gara: bandiere, safety car, virtual safety car                          | 179   | sapere in quali giri la gara era rallentata       |
| `drivers`      | numero, nome e sigla di ogni pilota                                                                | 22    | scrivere la sigla (HAM, OCO…) al posto del numero |

I 48 pit stop coincidono con il [riepilogo ufficiale della FIA](https://www.fia.com/sites/default/files/2026_07_esp_f1_r0_timing_racepitstopsummary_v01.pdf): stessi piloti, stessi giri, stesse durate.

### Il dataset finale

`data/dataset.csv` ha **una riga per ogni pilota alla fine di ogni giro**: 1.164 righe in tutto, di cui 48 sono pit stop. Esempio:

```csv
driver,lap_number,compound,SOFT,MEDIUM,HARD,tyre_age,stint_number,laps_remaining,pace_loss,neutralized,pit_next_lap
OCO,1,SOFT,1,0,0,1,1,65,0.0,0,0
OCO,2,SOFT,1,0,0,2,1,64,0.0,0,0
OCO,3,SOFT,1,0,0,3,1,63,0.53,0,0
```

Le colonne che il modello usa (**feature**) sono otto:

| Colonna                  | Significato                                                                                          |
|--------------------------|------------------------------------------------------------------------------------------------------|
| `SOFT`, `MEDIUM`, `HARD` | La gomma montata. Una colonna per mescola: vale 1 quella giusta, 0 le altre                          |
| `tyre_age`               | Quanti giri ha già fatto quella gomma                                                                |
| `stint_number`           | 1 = gomme di partenza, 2 = dopo la prima sosta, e così via                                           |
| `laps_remaining`         | Quanti giri mancano alla fine della gara                                                             |
| `pace_loss`              | Quanti secondi il pilota sta perdendo rispetto al suo giro migliore con quelle gomme: misura l'usura |
| `neutralized`            | 1 se nel giro c'è la safety car o la virtual safety car                                              |

La colonna da indovinare (**etichetta**) è `pit_next_lap`: **1** se il pilota rientra ai box nel giro successivo, **0** se resta in pista.

Le mescole sono divise in tre colonne invece di scrivere SOFT=1, MEDIUM=2, HARD=3, perché altrimenti il modello penserebbe che la HARD "valga il triplo" della SOFT. Questa tecnica si chiama *one-hot encoding*.

## Architettura del modello

### Il punto di partenza: il perceptron

Il perceptron è un singolo "neurone" che lavora in tre passi:

1. **Moltiplica** ogni informazione per un peso, che ne indica l'importanza, e somma tutto
2. **Decide**: se la somma è positiva risponde 1 (pit), altrimenti 0 (no pit)
3. **Si corregge** solo quando sbaglia, spostando un po' i pesi verso la risposta giusta

È semplice, ma sa separare i casi solo con una "linea retta". La decisione del pit stop è più complicata: una gomma soft con 12 giri è da cambiare, una hard con 12 giri no, e a tre giri dalla fine nessuno si ferma anche con le gomme finite.

### La nostra rete: il Multilayer Perceptron

L'MLP mette **più neuroni in fila, su più strati**:

```
 8 informazioni        8 neuroni nascosti         1 uscita
 sul giro         →    (ognuno guarda i dati  →   probabilità
 (gomma, usura...)      a modo suo)                di pit stop
```

Rispetto al perceptron cambiano quattro cose.

1. **Lo strato nascosto.** Tra le informazioni in ingresso e la risposta finale ci sono 8 neuroni. Ognuno dà importanze diverse alle informazioni e si fa la sua opinione; poi il neurone di uscita mette insieme le 8 opinioni.
2. **Le funzioni di attivazione.** I neuroni nascosti usano la *ReLU*: tengono i valori positivi e azzerano quelli negativi, cioè "parlano solo se sono convinti". È questo che permette alla rete di tracciare confini "piegati" invece di una sola retta. Il neurone di uscita usa la *sigmoide*, che trasforma il risultato in una **probabilità tra 0 e 1** invece di un sì/no secco.
3. **La misura dell'errore (loss).** Il modello non conta solo *se* ha sbagliato, ma *quanto*. Dire "1%" a un pilota che poi si ferma è un errore molto più grave che dire "40%". Si usa la *binary cross-entropy*.
4. **La backpropagation.** Dopo ogni errore la rete risale all'indietro, dall'uscita verso l'ingresso, e calcola quanto ogni peso ha contribuito allo sbaglio. Poi corregge tutti i pesi di un piccolo passo (*discesa del gradiente*). Il perceptron si corregge solo quando sbaglia del tutto; l'MLP si corregge sempre, in proporzione all'errore.

In tutto la rete ha **81 numeri**: 64 pesi tra ingresso e strato nascosto, 8 bias dello strato nascosto, 8 pesi verso l'uscita e 1 bias dell'uscita. Addestrarla significa trovare il valore giusto di questi 81 numeri.

### Il problema dei pit stop rari

Su 1.164 giri, i pit stop sono solo 48 (4%). Un modello "pigro" che risponde sempre "nessuno si ferma" avrebbe ragione il 96% delle volte, senza trovare nemmeno una sosta. Per evitarlo, durante l'addestramento **ogni pit stop pesa circa come 22 giri normali**, come un professore che dà più punti alle domande difficili.

### Come viene fatto l'addestramento

1. **Divisione per pilota**: 16 piloti servono per allenare il modello, 6 per metterlo alla prova: Hamilton, Antonelli, Leclerc, Verstappen, Norris e Russell. Non si mescolano i giri a caso: due giri consecutivi dello stesso pilota sono quasi identici, e se uno finisse nell'allenamento e l'altro nel test il modello sembrerebbe più bravo di quanto è. I 6 piloti di test hanno finito tutti la gara, quindi i loro giri sono 367 su 1.164, circa un terzo del totale.
2. **Standardizzazione**: le informazioni vengono messe sulla stessa scala (i giri mancanti arrivano a 65, le colonne delle mescole valgono solo 0 o 1). Le medie si calcolano solo sui piloti di allenamento, per non "far vedere le risposte" al modello.
3. **Allenamento**: la rete ripassa i dati 1.000 volte, sbagliando un po' meno ogni volta.
4. **Decisione**: se la probabilità prevista è almeno 0,5, il modello dice "rientra ai box".

## Come funziona il codice, file per file

Questa sezione spiega cosa fa ogni parte del codice. I numeri di riga sono quelli che si vedono aprendo il file in Visual Studio Code.

### download_data.py – scaricare i dati

| Righe | Cosa fa                                                                                                                             |
|-------|-------------------------------------------------------------------------------------------------------------------------------------|
| 1     | Una descrizione del file, che Python non esegue.                                                                                    |
| 3–7   | Carica le librerie: `json` per i file di dati, `os` per creare cartelle, `time` per le pause, `requests` per scaricare da internet. |
| 9     | `SESSION_KEY = 11307` è il codice con cui OpenF1 identifica la gara di Barcellona 2026.                                             |
| 10    | L'elenco delle cinque tabelle da scaricare.                                                                                         |
| 12    | Crea la cartella `data/raw`, se non esiste già.                                                                                     |
| 14    | Un ciclo: le righe seguenti vengono ripetute per ognuna delle cinque tabelle.                                                       |
| 15    | Costruisce l'indirizzo internet della tabella, per esempio `https://api.openf1.org/v1/laps?session_key=11307`.                      |
| 16    | Scarica la tabella e la trasforma in dati che Python può usare.                                                                     |
| 17–18 | Salva i dati in un file, per esempio `data/raw/laps.json`.                                                                          |
| 19    | Stampa quante righe sono arrivate, come controllo.                                                                                  |
| 20    | Aspetta un secondo: OpenF1 accetta poche richieste al secondo.                                                                      |

### build_dataset.py – costruire la tabella per il modello

| Righe | Cosa fa                                                                                                                                                                                                                                  |
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 6–8   | Carica `json` per leggere i file e `pandas` per lavorare con le tabelle.                                                                                                                                                                 |
| 10–11 | `FEATURES`: l'elenco delle 8 colonne che il modello userà.                                                                                                                                                                               |
| 12    | `LABEL`: il nome della colonna da indovinare, `pit_next_lap`.                                                                                                                                                                            |
| 13    | `TOTAL_LAPS = 66`: i giri della gara.                                                                                                                                                                                                    |
| 16–18 | La funzione `load`: apre un file della cartella `data/raw` e lo trasforma in una tabella.                                                                                                                                                |
| 22    | Carica i giri, tenendo solo pilota, numero del giro e tempo del giro.                                                                                                                                                                    |
| 23–26 | Carica le altre quattro tabelle.                                                                                                                                                                                                         |
| 29    | Affianca a ogni giro le informazioni sulle gomme dello stesso pilota.                                                                                                                                                                    |
| 30    | Tiene solo il treno di gomme giusto per ogni giro (quello in cui quel giro cade davvero).                                                                                                                                                |
| 31    | Calcola l'età della gomma: i giri che aveva quando è stata montata più quelli fatti in gara. Esempio: le soft di Hamilton avevano già 2 giri, quindi a fine giro 10 ne hanno 12.                                                         |
| 34–35 | Trasforma la mescola in tre colonne di 0 e 1 (one-hot encoding).                                                                                                                                                                         |
| 38    | Calcola i giri che mancano alla fine: 66 meno il giro attuale.                                                                                                                                                                           |
| 41    | Mette in ordine i giri, pilota per pilota.                                                                                                                                                                                               |
| 42    | Per ogni pilota e ogni treno di gomme, trova il giro più veloce fatto fino a quel momento.                                                                                                                                               |
| 43    | `pace_loss`: quanti secondi il giro attuale è più lento di quel giro migliore. Se cresce, la gomma si sta consumando. Il valore viene bloccato a 10 secondi, perché i giri sotto safety car sono molto più lenti e falserebbero il dato. |
| 46    | Tra i messaggi della direzione gara tiene solo quelli sulla safety car.                                                                                                                                                                  |
| 47–48 | Trova i giri in cui la safety car inizia ("DEPLOYED") e finisce ("ENDING"). A Barcellona: giri 41–42 e 63–65.                                                                                                                            |
| 49    | Elenca tutti i giri con la gara rallentata: 41, 42, 63, 64, 65.                                                                                                                                                                          |
| 50    | La colonna `neutralized` vale 1 in quei giri, 0 negli altri.                                                                                                                                                                             |
| 53    | Crea l'elenco di tutti i pit stop come coppie (pilota, giro). Hamilton, numero 44, si è fermato al giro 11: la coppia è (44, 11).                                                                                                        |
| 54    | **L'etichetta**: per ogni riga controlla se c'è un pit stop **al giro successivo**. A fine giro 10 di Hamilton cerca (44, 11), la trova, quindi l'etichetta vale 1.                                                                      |
| 57    | Segna i giri in cui il pilota entra ai box: lì è già in corsia box, non c'è niente da prevedere.                                                                                                                                         |
| 58    | Trova l'ultimo giro di ogni pilota.                                                                                                                                                                                                      |
| 59    | Toglie i giri di rientro e l'ultimo giro di ogni pilota. Si passa da 1.241 giri a 1.164 righe.                                                                                                                                           |
| 62    | Scrive la sigla del pilota al posto del numero (44 diventa HAM).                                                                                                                                                                         |
| 63    | Restituisce solo le colonne utili.                                                                                                                                                                                                       |
| 66–69 | Quando si lancia il file: costruisce la tabella, la salva in `data/dataset.csv` e stampa il riepilogo.                                                                                                                                   |

### perceptron_classifier.py – il modello semplice

| Righe | Cosa fa                                                                                                                                                               |
|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1–4   | La descrizione del file. |
| 6–16  | Carica le librerie e i nostri file: l'elenco delle colonne e i due modelli. |
| 18–19 | **I 6 piloti di test**, scritti a mano: Hamilton, Antonelli, Leclerc, Verstappen, Norris e Russell. |
| 22    | La funzione `evaluate`, che stampa i risultati di un modello. |
| 23–25 | Calcola precision, recall e F1 considerando **pit e no pit insieme**, con la media macro (vedi *Risultati*). |
| 26–30 | Stampa accuracy, precision, recall e F1. |
| 31–32 | Stampa la confusion matrix. |
| 35–41 | Legge le impostazioni scritte nel terminale (es. `--epochs 2000`); se non si scrive niente usa i valori di default. |
| 44–46 | Carica il dataset: `X` sono le 8 informazioni, `y` le risposte giuste. |
| 49–50 | **Divide per pilota**: i giri dei 6 piloti di test da una parte, quelli degli altri 16 dall'altra. Tutti i giri di uno stesso pilota stanno dalla stessa parte. |
| 51    | Stampa i piloti del test. |
| 54–55 | **Standardizza**: mette tutte le informazioni sulla stessa scala, usando le medie dei soli piloti di allenamento. |
| 58–59 | Addestra il perceptron e lo valuta sui piloti di test. |
| 62    | Calcola quanto deve pesare un pit stop: giri normali diviso pit stop: 763 / 34, circa 22. |
| 63–66 | Crea e addestra l'MLP. |
| 67    | Valuta l'MLP sui piloti di test. |
| 70–72 | Salva il modello e lo scaler: servono a `predict.py`. |
| 75–80 | Disegna e salva il grafico della loss (`loss_curve.png`). |

### check_separability.py – i dati si dividono con una retta?

| Righe | Cosa fa                                                                                                                                                                                              |
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1–10  | La descrizione del file.                                                                                                                                                                             |
| 12–21 | Carica le librerie e i due modelli.                                                                                                                                                                  |
| 24–26 | Carica **tutti** i giri, pit e no pit insieme, e li mette sulla stessa scala. Qui non si divide in allenamento e test: la separabilità è una proprietà dei dati, non del modello.                    |
| 27    | Stampa quanti pit e quanti no pit ci sono.                                                                                                                                                           |
| 30–31 | Cerca giri con informazioni identiche ma risposte diverse. Se ci fossero, nessun modello potrebbe separarli. Il risultato è 0.                                                                       |
| 35–40 | **Il test**: chiede alla programmazione lineare (un metodo matematico) se esiste una retta che mette tutti i pit da una parte e tutti i no pit dall'altra. Se la trova stampa "SI", altrimenti "NO". |
| 43–48 | Prepara tre modelli da confrontare: il perceptron, la migliore retta possibile (una SVM lineare) e l'MLP.                                                                                            |
| 50–55 | Li addestra uno alla volta su tutti i dati e stampa quanti pit trovano e quanti falsi allarmi fanno.                                                                                                 |

### predict.py – chiedere al modello

| Righe | Cosa fa                                                                                                                        |
|-------|--------------------------------------------------------------------------------------------------------------------------------|
| 1–7   | La descrizione del file, con un esempio d'uso.                                                                                 |
| 9–15  | Carica le librerie, l'elenco delle colonne e il modello.                                                                       |
| 17–20 | Legge dal terminale la sigla del pilota e il giro.                                                                             |
| 23–26 | Carica il modello addestrato, lo scaler e il dataset.                                                                          |
| 29    | Cerca la riga di quel pilota a quel giro.                                                                                      |
| 30–31 | Se non la trova (giro di rientro, ultimo giro o pilota ritirato), si ferma con un messaggio.                                   |
| 34    | **La previsione**: prende le 8 informazioni del giro, le mette sulla stessa scala dell'allenamento e le fa passare nella rete. |
| 35–39 | Stampa la situazione, la probabilità, la decisione e cosa è successo davvero.                                                  |

## Risultati

### Il confronto sui piloti di test

Sui 6 piloti di test (Hamilton, Antonelli, Leclerc, Verstappen, Norris e Russell), che insieme hanno fatto 14 pit stop:

| Modello             | Accuracy | Precision | Recall    | F1        | Pit stop trovati |
|---------------------|----------|-----------|-----------|-----------|------------------|
| Perceptron semplice | 0,954    | **0,680** | 0,667     | **0,673** | 5 su 14          |
| **MLP**             | 0,856    | 0,585     | **0,822** | 0,606     | **11 su 14**     |

Come leggere le metriche:

- **Accuracy**: la percentuale di risposte giuste su tutti i giri.
- **Precision**: quando il modello dà una risposta, quante volte ha ragione.
- **Recall**: dei casi veri, quanti il modello ne ha riconosciuti.
- **F1**: un voto unico che mette insieme precision e recall.

Precision, recall e F1 sono calcolate su **pit e no pit insieme**, con la *media macro*: si calcola il valore per ciascuna delle due classi e poi se ne fa la media, dando lo stesso peso a pit e no pit. Non usiamo la media pesata sul numero di giri, perché i no pit sono il 95% e coprirebbero del tutto i pit stop.

**Cosa ci dicono questi numeri:**

- **L'accuracy da sola inganna.** Il perceptron ha l'accuracy più alta (95%) perché dice quasi sempre "no pit", ma trova solo 5 soste su 14.
- **L'MLP trova molti più pit stop**: 11 su 14, su piloti che non ha mai visto. Per questo ha la recall più alta.
- **Il prezzo sono i falsi allarmi**: 50 volte l'MLP chiama un pit che non avviene, contro gli 8 del perceptron. Per questo la sua precision è più bassa, e in questo test il perceptron ha anche l'F1 un po' più alta. Quasi due falsi allarmi su cinque (19 su 50) cadono però nei due giri prima di una sosta vera: il modello capisce che la finestra si sta aprendo, ma con un po' di anticipo.
- **Dipende dai piloti scelti**: i piloti dei top team seguono strategie regolari, quindi anche il perceptron qui va meglio che su altri piloti. Con un gruppo di test diverso i numeri cambiano.

Il modello non è perfetto, e non è questo l'obiettivo: il progetto vuole mostrare in modo spiegabile come si passa dal perceptron a un MLP e perché il secondo serve.

### La confusion matrix

È la tabella che mostra, caso per caso, come ha risposto il modello. Per l'MLP:

```
[[303  50]
 [  3  11]]
```

La prima riga sono i 353 giri senza pit: 303 riconosciuti correttamente, 50 falsi allarmi. La seconda riga sono i 14 pit stop veri: 3 mancati, 11 trovati.

### I dati sono linearmente separabili?

`check_separability.py` risponde di **no**: nessuna retta mette tutti i pit da una parte e tutti i no pit dall'altra. I pit cadono alla fine di ogni treno di gomme, ma a età diverse (circa 12 giri nel primo stint, tra 15 e 23 nel secondo), e una retta sola non riesce a prenderli tutti. Non ci sono però giri identici con risposte diverse, quindi un modello non lineare può separarli meglio.

Su tutti i giri, pit e no pit insieme:

| Modello                                | Pit trovati | Falsi allarmi |
|----------------------------------------|-------------|---------------|
| Perceptron semplice                    | 10 su 48    | 29 su 1.116   |
| Migliore retta possibile (SVM lineare) | 41 su 48    | 302 su 1.116  |
| MLP (8 neuroni)                        | 41 su 48    | 170 su 1.116  |

A parità di pit trovati, l'MLP fa quasi la metà dei falsi allarmi della migliore retta: è la prova che lo strato nascosto serve. Questi numeri sono calcolati sugli stessi dati usati per allenare, quindi misurano quanto il modello riesce a *separare* i dati, non quanto bene *prevede* giri nuovi.

### La curva di apprendimento

Il file `loss_curve.png` mostra come l'errore dell'MLP scende durante l'addestramento: se la curva cala, la rete sta imparando.

## Esempi

### Esempio di addestramento

```
python train_mlp.py
```

Output (parte finale):

```
Righe: 1164  |  pit stop: 48
Piloti nel test: HAM, ANT, LEC, VER, NOR, RUS
...
Addestramento MLP...
Epoca 200: loss 0.7993
Epoca 400: loss 0.7388
Epoca 600: loss 0.7030
Epoca 800: loss 0.6829
Epoca 1000: loss 0.6673

===== MLP =====
Accuracy:  0.856
Precision: 0.585
Recall:    0.822
F1:        0.606
Confusion matrix (righe = reale, colonne = previsto):
[[303  50]
 [  3  11]]

Salvati: mlp_model.pkl, scaler.pkl, loss_curve.png
```

### Esempi di previsione

Tutti i piloti degli esempi fanno parte del test, quindi il modello non li ha mai visti durante l'addestramento.

**Una previsione giusta.**

```
python predict.py --driver HAM --lap 10
```

```
HAM, fine giro 10: gomma SOFT con 12 giri
Probabilita' di pit al giro 11: 0.87
Previsione: RIENTRA AI BOX
Realta':    e rientrato
```

**Un giro senza sosta, riconosciuto.**

```
python predict.py --driver HAM --lap 20
```

```
HAM, fine giro 20: gomma HARD con 9 giri
Probabilita' di pit al giro 21: 0.03
Previsione: RESTA IN PISTA
Realta':    non e rientrato
```

**Una sosta mancata di poco.** Con `python predict.py --driver HAM --lap 26` il modello dà il 46%, appena sotto la soglia del 50%, ma Hamilton è rientrato. Il modello aveva capito che la sosta era vicina, ma non abbastanza da dire "rientra".

**Un falso allarme.** Con `python predict.py --driver RUS --lap 33` il modello dà il 97%, ma Russell non è rientrato al giro dopo: si è fermato due giri più tardi, al giro 36. Il modello aveva visto giusto, ma in anticipo.

## Risoluzione dei problemi

### Problemi comuni

1. **`command not found: python` o `pip`**
    - **Causa**: sul Mac i comandi si chiamano spesso `python3` e `pip3`
    - **Soluzione**: usare `python3` e `pip3` al posto di `python` e `pip`
2. **`ModuleNotFoundError: No module named 'numpy'`** (o pandas, sklearn, scipy...)
    - **Causa**: le librerie non sono installate nell'ambiente in uso
    - **Soluzione**: controllare che l'ambiente `f1-pitstop` sia attivo e lanciare `pip install -r requirements.txt`
3. **`FileNotFoundError: mlp_model.pkl`**
    - **Causa**: si sta provando a fare una previsione prima di aver addestrato il modello
    - **Soluzione**: lanciare prima `python train_mlp.py`
4. **`Giro non trovato`** in `predict.py`
    - **Causa**: quel giro non è nel dataset. Può essere un giro in cui il pilota era già ai box, l'ultimo giro del pilota o un giro dopo il suo ritiro
    - **Soluzione**: provare un altro giro o un altro pilota

### Consigli sulle impostazioni

1. **Learning rate**: valori troppo alti fanno "saltare" la rete senza farla imparare; tra 0,05 e 0,3 funziona bene
2. **Epoche**: se la curva della loss sta ancora scendendo alla fine, si possono aumentare
3. **Neuroni nascosti**: più neuroni rendono la rete più capace, ma con pochi dati rischia di imparare a memoria invece di capire

## Limiti

- **Una sola gara**: 48 pit stop sono pochi, quindi i risultati possono cambiare parecchio scegliendo piloti di test diversi.
- **Gli imprevisti non si possono prevedere**: se la safety car entra durante il giro successivo, o un pilota rientra per un guasto, alla fine del giro il modello non può saperlo.
- **Mancano informazioni strategiche**: distacchi dagli avversari, posizione in gara, strategie dei rivali, che i muretti box usano davvero.

## Sviluppi futuri

- **Più gare**: aggiungere altre gare del 2026 e mettere alla prova il modello su circuiti mai visti
- **Più informazioni**: distacchi dagli avversari e posizione in classifica (disponibili su OpenF1)
- **Condizioni di pioggia**: gomme intermedie e da bagnato
- **Scelta automatica della soglia**: invece di 0,5, scegliere la soglia che bilancia meglio pit trovati e falsi allarmi
- **Validazione incrociata**: ripetere l'esperimento più volte con gruppi di piloti diversi nel test, per risultati più affidabili

## Fonti dei dati

- [OpenF1](https://openf1.org/): dati di cronometraggio della Formula 1
- [FIA – Race Pit Stop Summary, Barcellona 2026](https://www.fia.com/sites/default/files/2026_07_esp_f1_r0_timing_racepitstopsummary_v01.pdf): usato per verificare i pit stop

Il progetto è stato sviluppato in *vibe coding*, cioè con l'aiuto di un assistente AI.

## Licenza

Questo progetto è open source e disponibile con licenza MIT (vedi il file `LICENSE`).

*I dati provengono da OpenF1, fonte non ufficiale, e dalla FIA, e sono usati solo a scopo didattico. F1, FORMULA 1 e i marchi collegati appartengono a Formula One Licensing B.V.*
