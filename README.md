# YuE2 Auto Vocal PRO — ComfyUI

Custom node autonomo per utilizzare i nodi ufficiali **YuE2** e **SheetSage2** dentro ComfyUI, senza WSL, servizi esterni, Seed-VC, RVC o conversione vocale post-generazione. Il ramo **Reference Voice** segue l'approccio `Ornn8/yue2-cover`: spartito trascritto dalla canzone sorgente, Lyrics nuove divise per sezioni, campione della voce di riferimento codificato come prefisso semantico nativo YuE2.

## Autore e riconoscimenti

**White Dragon (Francesco Durante)** ha realizzato la **reimplementazione ComfyUI di `yue2-cover` per la funzionalità Reference Voice**, integrandola nel workflow YuE2 Auto Vocal PRO. Questo ramo utilizza il meccanismo nativo YuE2 di condizionamento mediante prefisso semantico; non usa Seed-VC, RVC né conversione vocale esterna.

**Progetto originale di riferimento:** [`Ornn8/yue2-cover`](https://github.com/Ornn8/yue2-cover), a cui spettano i crediti per l'implementazione originale. Il parser e il compositore ABC riutilizzati sono conservati in `vendor/yue2_cover_main/` insieme alla relativa licenza Apache-2.0. La reimplementazione ComfyUI e il progetto originale sono componenti distinti; questa attribuzione non implica che l'autore del progetto originale abbia sviluppato o approvato il presente custom node.

## Installazione pulita

1. Chiudi ComfyUI. Sposta **fuori** da `custom_nodes/` ogni copia precedente di `ComfyUI-YuE2-Auto-Vocal-PRO-v2` (non lasciare due copie caricabili).
2. Estrai la cartella `ComfyUI-YuE2-Auto-Vocal-PRO-v2` da questo ZIP direttamente in `ComfyUI/custom_nodes/`.
3. I file `yue2_3b_bf16.safetensors`, SheetSage2, MERT-v2-FullSong, tokenizer head joint-v9 e LoRA NAR devono rimanere nelle cartelle modelli standard di ComfyUI; **non** sono inclusi nello ZIP.
4. Installa `requirements.txt` **solo se mancano dipendenze**, usando il Python già integrato in ComfyUI. Non reinstallare Torch/CUDA; non installare `yue2-infer`.
5. Riavvia ComfyUI, apri `YuE2_Cover_AutoVocal_PRO_V2.json` e scegli canzone sorgente e campione Reference Voice. Compila Lyrics e Style.

## Workflow incluso

Quattro aree, un solo workflow, con i titoli in inglese: **Inputs and Parameters**, **Music, ABC and Replacement Lyrics**, **Reference Voice**, **Output and Quality Control**. Il selettore finale è inizialmente su `reference_voice`. È disponibile anche il percorso `standard` per un confronto a parità di testo sorgente e spartito.

Il nodo `ABC + Lyrics · Alignment` riceve l'ABC prodotto da SheetSage2. Il testo scritto nel campo **Replacement Lyrics** è l'unico testo target da fornire ai generatori. Le etichette `[Verse]`, `[Chorus]` e simili orientano l'assegnazione; le sezioni strumentali prive di Lyrics non devono essere riempite con parole inventate dal planner. Una melodia valida con molte note e poche parole **non è rifiutata** per un rapporto sillabe/note.

`existing_vocal=preserve` conserva la traccia Vocal originale nello spartito; `overwrite` può convertire una parte strumentale compatibile in Vocal. Entrambe le opzioni, insieme a `section_policy`, `instrumental_handling` e `vocal_melody_mode`, sono propagate anche all'eventuale validatore del percorso ABC alternativo. Non cambiano il contenuto testuale delle Lyrics dell'utente.

`max_duration` è **un limite massimo**. Se il brano sorgente rientra nel limite, non viene accorciato in funzione del conteggio delle sillabe. Se supera il limite, il planner può scegliere un **estratto musicale contiguo e valido**, indicando nel report che non si tratta della cover integrale. Non altera le parole per farle rientrare nella durata. Se nessun estratto coerente è disponibile, viene segnalato un vincolo reale invece di creare una melodia arbitraria o ignorare il limite.

**Pronuncia e intelligibilità:** la fedeltà testuale (stesse parole in ingresso) non implica dizione corretta (sillabe articolate). Il report di pianificazione contiene `diction_advisory`: parole/s, sillabe stimate/s e sillabe stimate per attacco di nota, calcolate separatamente per sezione ABC. Se una sezione è troppo densa, viene segnalata come **pressione articolatoria potenziale** senza modificare, accorciare o inventare il testo. Il Fidelity Audit aggiunge `diction_review` quando Whisper è disponibile, ma non confonde i suoi errori di riconoscimento con una dimostrata pronuncia difettosa. Nessun filtro sul waveform può garantire una dizione corretta o sostituire l'ascolto.

**Audio vuoto:** l'Audit verifica che l'audio selezionato contenga campioni finiti e un segnale misurabile prima di consegnarlo al salvataggio. Un tensor realmente vuoto, NaN/Inf o silenzioso viene rifiutato; un brano con porzioni ampiamente silenziose viene segnalato mediante RMS su finestre campionate. `report_only` non disabilita la protezione contro l'audio completamente silenzioso.

**Overwrite e legature:** i gruppi collegati da legature su *entrambe* le voci sono considerati congiuntamente. Se la conversione non può preservare battute, tonalità, accordi e validità del parser, viene annullata per l'intero ABC e riportata come rollback: lo spartito sorgente resta valido, ma la conversione richiesta potrebbe non essere stata completata. Verificare l'avviso; non scambiare un rollback per un overwrite riuscito.

Il workflow incluso usa esattamente la disposizione del file fornito dall'utente (`YuE2_Cover_AutoVocal_PRO_V2(3).json`), conservandone posizioni, parametri musicali e collegamenti; tutti i titoli e i gruppi sono in inglese, i due `LoadAudio` e il campo Lyrics sono **vuoti** fino alla selezione dell'utente; il formato **MP3 di uscita** rimane intenzionalmente selezionato. I soli valori serializzati incoerenti nei widget dell'Audit e dell'Alignment sono normalizzati ai tipi richiesti dal codice, senza modificare la topologia.

Il workflow usa il **prefisso Reference Voice nativo di YuE2**, non sistemi esterni. Se il riferimento induce YuE2 a riprendere parole vecchie, si tratta di un problema di **fedeltà acustica** che non è risolvibile o escludibile con i soli controlli simbolici.

## Vocal Fidelity Audit

Il nodo `Final Audio · Vocal Fidelity Audit` è collegato **dopo** il selettore Standard/Reference Voice e prima del salvataggio. Analizza quindi l'**audio selezionato**, e il JSON dichiara quale percorso è stato auditato. `report_only` non blocca il salvataggio quando Whisper non è disponibile o la sua trascrizione è incerta; `fail_closed` è una scelta esplicita che può rifiutare la generazione per evidenze insufficienti. Whisper sui brani cantati è imperfetto: un audit favorevole non garantisce che ogni parola venga cantata correttamente.

## Diagnostica forense

Lo strumento è integrato nel custom node e crea checkpoint ZIP sotto:

`ComfyUI/output/yue2_forensic/`

Nel workflow consegnato `forensic_replay=true` è **attivo** nei due nodi principali per una prova diagnostica riproducibile. **Attenzione alla riservatezza:** in questa modalità gli ZIP locali contengono ABC, Lyrics, Style e copia del codice Python necessario al replay in chiaro; **non** contengono audio, tensori audio né pesi dei modelli. Prima di condividere lo ZIP, controlla le informazioni. Per usare soltanto hash e struttura degli input, imposta `forensic_replay=false` nei due nodi.

Quando un errore avviene prima della generazione audio, il checkpoint del nodo contiene già lo spartito, i parametri e gli eventuali errori. Se il fallimento avviene nel codice nativo di ComfyUI, rimane disponibile l'ultimo checkpoint del nostro plugin e va allegato anche il traceback della console. Gli hash dei sorgenti ufficiali locali possono essere confrontati con la versione pubblica, senza supporre che siano identici.

## Verifiche e limiti della distribuzione

La build è stata verificata senza pesi con matrice dei selettori, simulazione dei grafi, regressioni, stress a due seed e controlli di topologia, oltre a test del codice estratto dallo ZIP. Sono controlli del **flusso simbolico**, non dell'inferenza reale YuE2 sulla tua RTX 5090. Input impossibili possono comportare un rifiuto di dominio esplicito: l'obiettivo è evitare eccezioni *inattese*, non promettere che qualsiasi ABC sia musicalmente valido.

**Non è ancora certificata la fedeltà acustica delle Lyrics in Reference Voice**: serve almeno una generazione effettiva, ascolto, report dell'audit e, se necessario, ZIP forense.

## Codice sorgente e licenze

Il parser e il compositore ABC riutilizzati da `Ornn8/yue2-cover` sono vendorizzati in `vendor/yue2_cover_main/` con la licenza Apache-2.0 originale. I modelli YuE2, SheetSage2 e tokenizer hanno licenze proprie, non incluse nello ZIP. Consulta le condizioni dei modelli e detieni i diritti necessari per i brani e i campioni vocali utilizzati.
