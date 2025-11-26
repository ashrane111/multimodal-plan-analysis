import pandas as pd
import re

# --- PASTE THE MAPPING FUNCTION HERE ---
def map_label_to_ontology(raw_label):
    text = str(raw_label).lower().strip()
    text_clean = re.sub(r'[+/\.]', ' ', text) # "k+rt" -> "k rt"
    
    # 1. Filter Dimensions (Keep this strict)
    if re.search(r'\d+[\'\"xX]\d+', text) or re.search(r'\d+\s*sq', text):
        return "Background"
    
    # 2. MAPPING LOGIC (Aggressive Finnish Support)
    
    # --- BEDROOM (MH, Makuu) ---
    if "makuu" in text or "alkovi" in text or "bedroom" in text: return "Bedroom"
    # Matches MH, MH1, MH 2, M H
    if re.search(r'\bmh\s*\d*', text_clean): return "Bedroom" 
    if re.search(r'\b(bed|guest|suite|dorm|bunk)\b', text_clean): return "Bedroom"

    # --- KITCHEN (K, Keittiö) ---
    if "keitti" in text or "kitchen" in text or "cooking" in text: return "Kitchen"
    # Added: K, KT, KS, RT (Dining area inside kitchen), P.K (Pienkeittiö)
    if re.search(r'\b(k|kt|ks|rt|ruok|dining|pantry|cook|nook|apuk|avok|pk)\b', text_clean): return "Kitchen"

    # --- DINING (Ruokailu - Separate class if you want, or map to Kitchen) ---
    # You had 46 "RUOKAILU" in background. Let's map them!
    if "ruokailu" in text or "dining" in text: return "Dining"

    # --- BATHROOM (WC, Pesu, Kylpy) ---
    if "kylpy" in text or "pesu" in text or "sauna" in text or "bath" in text or "toilet" in text: return "Bathroom"
    if re.search(r'\b(wc|kph|ph|sh|psh|kh|s|shower|restroom|powder)\b', text_clean): return "Bathroom"
    if "puku" in text or "pkh" in text: return "Bathroom" # Dressing room usually inside bath

    # --- LIVING ROOM (OH, Olohuone, Oleskelu) ---
    if "olohuone" in text or "living" in text or "lounge" in text or "tupa" in text: return "LivingRoom"
    # Added: OLESKELU (Lounge/Living)
    if "oleskelu" in text: return "LivingRoom"
    if re.search(r'\b(oh|family|great|media|rec|ask|ark|salon|takka)\b', text_clean): return "LivingRoom"

    # --- STORAGE (Varasto, VH, Ullakko) ---
    if "varasto" in text or "vaate" in text or "storage" in text or "closet" in text: return "Storage"
    # Added: ULLAKKO (Attic), KELLARI (Basement), KYLMIÖ (Cold Storage)
    if any(x in text for x in ["kellari", "kylmiö", "tekn", "komero", "ullakko", "katt"]): return "Storage"
    if re.search(r'\b(vh|sk|kom|var|walk-in|wic|wardrobe|utility|laundry|khh|pannu|ljh|öljy|lämm|kuiv|kell|kylm)\b', text_clean): return "Storage"

    # --- OUTDOOR ---
    if "parveke" in text or "terassi" in text or "veranta" in text or "kuisti" in text: return "Outdoor"
    if re.search(r'\b(out|deck|porch|terrace|balcony|patio|piha|pergola|vilpola|lasitettu)\b', text_clean): return "Outdoor"

    # --- GARAGE ---
    if "auto" in text or "garage" in text: return "Garage"
    if re.search(r'\b(car|parking|at|katos|vaja|tall)\b', text_clean): return "Garage"

    # --- ENTRY (Eteinen, Tuulikaappi, Kuraeteinen) ---
    if "eteinen" in text or "entry" in text or "hall" in text or "aula" in text: return "Entry"
    # Added: KURAET (Mudroom)
    if "kura" in text: return "Entry"
    if "käytävä" in text or "corridor" in text: return "Hallway"
    if re.search(r'\b(et|tk|lobby|foyer|tuuli|pr|tkh)\b', text_clean): return "Entry"

    # --- OFFICE ---
    if "työ" in text or "office" in text or "study" in text: return "Office"
    if re.search(r'\b(den|library|work|kirjasto|tyoh)\b', text_clean): return "Office"
    
    # --- STAIRS/VOID ---
    if "parvi" in text or "loft" in text: return "Bedroom" # Sleeping loft often counted as bedroom space
    if "porras" in text or "stair" in text: return "Stairs"
    if re.search(r'\b(utility|laundry|khh|tekn|pannu|pannuh|ljh|öljy|boiler|mud)\b', text_clean): return "Utility"
    if re.search(r'\b(closet|storage|walk-in|wardrobe|var|varasto|vh|vaate|kom|kell|ullakko|säilytys|sk|kylmiö|kylm|kyl|kuiv|lämm|kylmä|puuvar)\b', text_clean): return "Storage"
    
    return "Background"
# --- MAIN ANALYSIS ---
def analyze():
    print("Loading CSV...")
    df = pd.read_csv("data/processed/parsed_layout.csv")
    
    print(f"Total Raw Rows: {len(df)}")
    
    # Apply Mapping
    df['mapped_class'] = df['label'].apply(map_label_to_ontology)
    
    # Get Counts
    counts = df['mapped_class'].value_counts()
    percentages = df['mapped_class'].value_counts(normalize=True) * 100
    
    print("\n--- CLASS DISTRIBUTION ---")
    print(f"{'Class':<15} | {'Count':<8} | {'Percent':<8}")
    print("-" * 35)
    for cls, count in counts.items():
        perc = percentages[cls]
        print(f"{cls:<15} | {count:<8} | {perc:.1f}%")

    # Check what is getting dumped into Background
    print("\n--- TOP 20 LABELS MAPPED TO BACKGROUND ---")
    backgrounds = df[df['mapped_class'] == "Background"]
    print(backgrounds['label'].value_counts().head(20))

if __name__ == "__main__":
    analyze()