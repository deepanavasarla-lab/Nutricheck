"""
Rule-based safety net for the 6 clinical entities (pure Python, no spaCy).

WHY THIS EXISTS
The statistical NER model was trained from scratch on a small, SYMPTOM-heavy dataset, so on new text it
often returns only SYMPTOM spans. This module finds the other categories with a medical lexicon + suffix
rules + context cues, and main.py merges the result with what the model found (the model always wins on
overlaps, the lexicon only ADDS what the model missed).

Returns spans: (start, end, LABEL) with offsets into the ORIGINAL text.
"""
import re
import unicodedata


def norm_same_length(text: str) -> str:
    """Lowercase + strip accents without changing the length (offsets stay valid)."""
    out = []
    for ch in text:
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)).lower()
        out.append(base if len(base) == 1 else (ch.lower()[:1] or " "))
    return "".join(out)


# ------------------------------- lexicons (accent-free, lowercase) -------------------------------
SYMPTOMS = """
febre|febril|tosse|tosse seca|tosse produtiva|vomitos|vomito|nauseas|nausea|diarreia|obstipacao|cansaco|astenia|
fadiga|cefaleia|cefaleias|tonturas|tontura|vertigens|vertigem|dispneia|falta de ar|edema|edemas|prurido|exantema|
erupcao cutanea|urticaria|dormencia|formigueiro|palpitacoes|calafrios|sudorese|suores|anorexia|insonia|sonolencia|
confusao|convulsoes|sincope|hemoptises|odinofagia|disfagia|epistaxis|inchaco|fraqueza|rigidez|congestao nasal|
rinorreia|espirros|mal estar|mal-estar|perda de peso|perda de apetite|perda de consciencia|perda de forca|
aumento de peso|pernas inchadas|visao turva|dificuldade respiratoria|dificuldade em respirar|sibilancia|
sibilos|palidez|taquicardia|bradicardia|hipotensao|dor|dores|mialgias|artralgias|lombalgia|pirose|azia|
colica|colicas|desidratacao|hematuria|disuria|poliuria|sangue nas fezes|sangue na urina|batimentos cardiacos acelerados
""".replace("\n", "").split("|")

TREATMENTS = """
antibiotico|antibioticos|antibioterapia|analgesico|analgesicos|anti-inflamatorio|anti-inflamatorios|antipiretico|
antipireticos|anticoagulante|anticoagulantes|anticoagulacao|anti-hipertensor|anti-hipertensores|diuretico|diureticos|
broncodilatador|broncodilatadores|corticoide|corticoides|corticosteroide|corticosteroides|insulinoterapia|
fluidoterapia|soroterapia|hidratacao|hidratacao endovenosa|oxigenoterapia|oxigenio|fisioterapia|quimioterapia|
radioterapia|hemodialise|dialise|cirurgia|intervencao cirurgica|sutura|drenagem|transfusao|reabilitacao|
apendicectomia|colecistectomia|ventilacao mecanica|ventilacao nao invasiva|vacinacao|internamento|repouso|dieta|
cuidados intensivos|antiemetico|antiemeticos|antiacido|antiacidos|antialergico|antihistaminico|anti-histaminico|
antifungico|antiviral|antivirais|tratamento sintomatico|tratamento conservador|
analgesia|imobilizacao|pensos|penso|curativo|inalador|nebulizacao|nebulizacoes|soro
""".replace("\n", "").split("|")

EXAMS = """
analises|analises clinicas|analises sanguineas|exames laboratoriais|exame laboratorial|exames complementares|
exame objetivo|exame neurologico|exame fisico|exame|exames|hemograma|hemograma completo|ecg|eletrocardiograma|
electrocardiograma|raio x|raio-x|radiografia|radiografia do torax|raio-x do torax|raio x do torax|rx|rx do torax|rx torax|tac|tc|
tac cerebral|tomografia computorizada|tomografia computadorizada|ressonancia magnetica|rm|ecografia|
ecografia abdominal|ecocardiograma|endoscopia|colonoscopia|gasimetria|gasometria|glicemia|glicemia capilar|
sumaria de urina|urina tipo ii|cultura de urina|urocultura|hemoculturas|hemocultura|proteina c reativa|pcr|
auscultacao cardiopulmonar|auscultacao|bioquimica|funcao renal|funcao hepatica|ionograma|eletroencefalograma|
espirometria|mamografia|biopsia|angiografia|cintigrafia|doppler|teste rapido|testes alergicos|exame de urina|
analise de sangue|analises de sangue|medicao da tensao arterial|tensao arterial|saturacao de oxigenio|
perfil lipidico|colesterol total|creatinina|ureia|troponina|inr|tsh
""".replace("\n", "").split("|")

# chronic / background diseases -> CONDITION by default
CONDITIONS = """
hipertensao|hipertensao arterial|hta|diabetes|diabetes mellitus|diabetes mellitus tipo 2|diabetes mellitus tipo 1|
diabetes tipo 2|diabetes tipo 1|dm|dm2|asma|dpoc|doenca pulmonar obstrutiva cronica|insuficiencia cardiaca|
insuficiencia cardiaca cronica|insuficiencia renal cronica|doenca renal cronica|dislipidemia|obesidade|
hipotiroidismo|hipertiroidismo|epilepsia|demencia|depressao|ansiedade|artrose|osteoporose|gota|enxaqueca|
fibrilhacao auricular|cardiopatia|cardiopatia isquemica|avc previo|doenca de crohn|esclerose multipla|
hipercolesterolemia|anemia cronica|cirrose|insuficiencia venosa|doenca celiaca|lupus|psoriase|
hiperplasia benigna da prostata|apneia do sono|tabagismo|alcoolismo|gravidez
""".replace("\n", "").split("|")

# acute diagnoses -> DIAGNOSIS by default
DIAGNOSES = """
gastrite|gastroenterite|pneumonia|bronquite|apendicite|colecistite|pancreatite|hepatite|infecao urinaria|
infecao respiratoria|infecao das vias respiratorias|infecao|sepsis|choque septico|litiase renal|fratura|
acidente vascular cerebral|avc|enfarte agudo do miocardio|enfarte do miocardio|enfarte|embolia pulmonar|trombose|
trombose venosa profunda|insuficiencia respiratoria|insuficiencia renal|insuficiencia renal aguda|
insuficiencia cardiaca aguda|cancro|cancro da mama|cancro do pulmao|cancro do colon|neoplasia|tumor|
anemia|desidratacao grave|rinite|sinusite|otite|amigdalite|faringite|dermatite|colite|gripe|tuberculose|
refluxo gastroesofagico|hemorragia|hemorragia digestiva|ulcera|ulcera gastrica|crise asmatica|laringite|
conjuntivite|celulite|abcesso|apendicite aguda|diverticulite|hernia|hernia discal|luxacao|entorse|covid-19|covid
""".replace("\n", "").split("|")

# common drugs (Portuguese spelling, accent-free) -- suffix rules below catch many more
MEDICATIONS = """
paracetamol|ibuprofeno|diclofenac|naproxeno|aspirina|acido acetilsalicilico|aas|tramadol|morfina|codeina|
amoxicilina|amoxicilina/acido clavulanico|ampicilina|penicilina|penicilinas|cefalexina|cefazolina|ceftriaxona|
azitromicina|claritromicina|eritromicina|ciprofloxacina|levofloxacina|gentamicina|vancomicina|metronidazol|
fluconazol|aciclovir|sulfametoxazol|trimetoprim|doxiciclina|tetraciclina|clindamicina|
metformina|insulina|gliclazida|glimepirida|sitagliptina|omeprazol|pantoprazol|esomeprazol|ranitidina|
furosemida|espironolactona|hidroclorotiazida|amlodipina|enalapril|captopril|lisinopril|losartan|valsartan|
bisoprolol|metoprolol|atenolol|carvedilol|digoxina|varfarina|heparina|enoxaparina|clopidogrel|rivaroxabano|
apixabano|atorvastatina|sinvastatina|rosuvastatina|prednisolona|prednisona|dexametasona|hidrocortisona|
salbutamol|budesonida|fluticasona|montelucaste|loratadina|cetirizina|desloratadina|levotiroxina|
sertralina|fluoxetina|escitalopram|citalopram|venlafaxina|diazepam|lorazepam|alprazolam|zolpidem|
carbamazepina|fenitoina|valproato|levetiracetam|metoclopramida|domperidona|ondansetrona|loperamida|
buscopan|butilescopolamina|dipirona|metamizol|nimesulida|celecoxib|colchicina|alopurinol|tamsulosina|
nitrofurantoina|cotrimoxazol|ranitidina|sulfasalazina|metotrexato|adrenalina|epinefrina|atropina|
oseltamivir|ivermectina|albendazol|mebendazol|vitamina d|ferro|acido folico|potassio|cloreto de potassio
""".replace("\n", "").split("|")

_MED_SUFFIXES = (
    "cilina", "micina", "floxacina", "oxacina", "cefalexina", "azol", "azolam", "pril", "sartan", "olol",
    "dipina", "statina", "prazol", "tidina", "formina", "parina", "profeno", "fenaco", "oxicam", "triptano",
    "setron", "peridona", "zepam", "oxetina", "tralina", "lucaste", "navir", "vudina", "cilina", "mide",
    "tiazida", "gliptina", "gliflozina", "glitazona", "caina", "barbital", "terol", "sonida", "solona",
)
_DISEASE_SUFFIXES = ("ite", "ose", "emia", "patia", "oma")
_DISEASE_STOP = {"dose", "doses", "visite", "coma", "diploma", "idioma", "sintoma", "sintomas", "glicemia",
                 "hemograma", "hematoma", "glaucoma", "aroma", "poema", "problema", "problemas", "sistema",
                 "programa", "tema", "plasma", "carcinoma", "osteoma", "diagnose", "dose", "prose", "pose",
                 "dermatose", "composta", "hipotensao", "creatinina", "anemia", "colesterolemia", "glicemia",
                 "bioquimica", "uremia", "hipoglicemia", "hiperglicemia", "ionograma"}
_DISEASE_STOP -= {"anemia"}
_STOP_AS_DRUG = {"normal", "mensal", "mide", "bemol", "abdomen", "paciente", "medico", "doente", "tratamento",
                 "hospital", "urgencia", "sexo", "masculino", "feminino", "antes", "durante", "depois"}

_PREP = {"de", "do", "da", "dos", "das", "no", "na", "nos", "nas", "em", "ao", "a", "por"}
_CONNECT_NEXT = {"de", "do", "da", "dos", "das"}
_ADJ_END = ("al", "ica", "ico", "icas", "icos", "ar", "ares", "ria", "rio", "osa", "oso", "ente", "inal")
_NEGATORS = {"sem", "nega", "negou", "negam", "ausencia", "nenhum", "nenhuma", "excluiu", "exclui", "afasta"}
_NEG_PHRASES = (("nao", "apresenta"), ("nao", "refere"), ("nao", "tem"), ("nao", "tinha"), ("nao", "houve"),
                ("nao", "se", "observou"), ("sem", "queixas", "de"))
_BACKGROUND = ("antecedentes", "antecedente", "historia", "conhecido", "conhecida", "portador", "portadora",
               "sob", "medicado", "medicada", "seguido", "seguida", "previo", "previa", "cronic", "hta", "ap")
_DIAG_CUES = ("diagnostic", "diagnos", "compativel", "suspeita", "dx", "quadro de", "admitid", "internad",
              "confirmou", "confirmad", "revelou", "evidenci", "sugestiv")

_TOKEN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


def _build_phrase_index():
    idx = {}
    # lower priority first, higher priority later (later assignments overwrite)
    for label, words in (("MEDICATION", MEDICATIONS), ("DIAGNOSIS", DIAGNOSES), ("CONDITION", CONDITIONS),
                         ("SYMPTOM", SYMPTOMS), ("TREATMENT", TREATMENTS), ("EXAM", EXAMS)):
        for w in words:
            w = w.strip()
            if not w:
                continue
            idx[tuple(_TOKEN.findall(w))] = label
    # "tensao arterial alta" is a symptom-ish finding, "hta" is a condition: keep as defined
    return idx


_PHRASES = _build_phrase_index()
_MAX_LEN = max(len(k) for k in _PHRASES)


def _is_negated(tokens, i, norm):
    """Is token i inside a negation scope (same clause, within 4 tokens before)?"""
    window = []
    k = i - 1
    while k >= 0 and len(window) < 4:
        # stop at a clause break ('.', ';', newline) between token k and the following token
        if any(ch in ".;\n" for ch in norm[tokens[k][2]: tokens[k + 1][1]]):
            break
        window.insert(0, tokens[k][0])
        k -= 1
    if any(w in _NEGATORS for w in window):
        return True
    for ph in _NEG_PHRASES:
        n = len(ph)
        for k in range(len(window) - n + 1):
            if tuple(window[k:k + n]) == ph:
                return True
    return False


def _clause_break(norm, a, b):
    return any(ch in ".;\n" for ch in norm[a:b])


def _context_label(norm, start, default):
    """CONDITION vs DIAGNOSIS for disease terms, decided by the words just before the span."""
    before = norm[max(0, start - 70): start]
    before = re.split(r"[.;\n]", before)[-1]
    if any(c in before for c in _DIAG_CUES):
        return "DIAGNOSIS"
    if any(c in before for c in _BACKGROUND):
        return "CONDITION"
    return default


def find_lexicon_entities(text: str):
    """-> sorted list of (start, end, LABEL) with offsets into the original text."""
    norm = norm_same_length(text)
    tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(norm)]
    spans = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = None
        for L in range(min(_MAX_LEN, n - i), 0, -1):
            key = tuple(t[0] for t in tokens[i:i + L])
            if key in _PHRASES:
                matched = (L, _PHRASES[key])
                break
        start_i, end_i, label = i, i, None
        if matched:
            L, label = matched
            end_i = i + L - 1
            word = tokens[i][0]
            # "dor" / "dores" + (prep + word | adjective)  ->  "dor abdominal", "dor no peito", "dor de cabeca"
            if word in ("dor", "dores") and L == 1:
                j = i + 1
                if j < n and tokens[j][0] in _PREP and j + 1 < n and not _clause_break(norm, tokens[end_i][2], tokens[j + 1][1]):
                    end_i = j + 1
                    if tokens[end_i][0] in ("o", "a", "os", "as") and end_i + 1 < n:
                        end_i += 1
                elif j < n and tokens[j][0].endswith(_ADJ_END) and not _clause_break(norm, tokens[end_i][2], tokens[j][1]):
                    end_i = j
            elif label in ("TREATMENT", "EXAM") and L == 1:
                # "tac cerebral", "exame neurologico", "cirurgia ..." keep a following adjective for exams
                j = i + 1
                if label == "EXAM" and j < n and tokens[j][0].endswith(_ADJ_END) and tokens[j][0] not in _PREP \
                        and not _clause_break(norm, tokens[end_i][2], tokens[j][1]) and word in ("exame", "exames", "analises", "ecografia", "radiografia", "tac", "tc"):
                    end_i = j
        else:
            w, w_start, w_end = tokens[i]
            nxt = tokens[i + 1][0] if i + 1 < n else ""
            nxt2 = tokens[i + 2][0] if i + 2 < n else ""
            alpha = w.isalpha()
            if alpha and len(w) >= 7 and w not in _STOP_AS_DRUG and w.endswith(_MED_SUFFIXES):
                label = "MEDICATION"
            elif alpha and len(w) >= 7 and w.endswith(_DISEASE_SUFFIXES) and w not in _DISEASE_STOP:
                label = "DIAGNOSIS"
            elif alpha and len(w) >= 5 and w not in _STOP_AS_DRUG and (
                    re.fullmatch(r"\d+(?:[.,]\d+)?", nxt) and nxt2 in ("mg", "g", "ml", "mcg", "ui", "u")
                    or nxt in ("mg", "mcg", "ml")) and w not in ("doses", "dose"):
                label = "MEDICATION"
        if label is None:
            i += 1
            continue

        s, e = tokens[start_i][1], tokens[end_i][2]
        # negation: "sem febre", "nega dor", "nao refere vomitos" -> not a finding (symptoms/diagnoses only)
        if label in ("SYMPTOM", "DIAGNOSIS", "CONDITION") and _is_negated(tokens, start_i, norm):
            i = end_i + 1
            continue
        if label in ("DIAGNOSIS", "CONDITION"):
            label = _context_label(norm, s, label)
        # the lone word "tratamento" / "exame(s)" is only worth keeping when it names something
        spans.append((s, e, label))
        i = end_i + 1
    return spans
