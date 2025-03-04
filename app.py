API_KEY2 = "AIzaSyAWfT7LegzDMdwebnghQu3vjqYBZfJUdo0"
API_METEO = "bd715056584f453cc48cb17af2e5e0bd"
import joblib
import numpy as np
import streamlit as st
import requests

# -------------------------------------------------------------------------
#                       PARAMÈTRES ET CHARGEMENT DU MODÈLE
# -------------------------------------------------------------------------
API_KEY = API_KEY2
modele = joblib.load("mon_modele.joblib")

# Tarif fixe (0.70 $/km) pour le modèle
TARIF_PAR_KM = 0.70

st.title("Prédiction du prix d'une course")


# -------------------------------------------------------------------------
#                       FONCTIONS D'APPEL D'API
# -------------------------------------------------------------------------
def get_address_suggestions(input_text, api_key):
    """Autocomplete des adresses (Google Places)."""
    if not input_text:
        return []

    url = (
        "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        f"?input={input_text}"
        f"&key={api_key}"
        "&types=geocode"
        "&language=fr"
    )
    resp = requests.get(url).json()
    #print(resp)
    suggestions = []
    if resp["status"] == "OK":
        for pred in resp["predictions"]:
            suggestions.append({
                "description": pred["description"],
                "place_id": pred["place_id"]
            })
        for i in suggestions:
            print(f"Adresse : {i['description']}\n"
                  f"ID : {i['place_id']} ")
    print(suggestions)
    return suggestions


def get_place_details(place_id, api_key):
    """Récupère (lat, lng) d’un lieu via Place Details API."""
    url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}"
        f"&key={api_key}"
    )
    resp = requests.get(url).json()
    if resp["status"] == "OK":
        loc = resp["result"]["geometry"]["location"]
        #print(loc)
        return loc["lat"], loc["lng"]
    else:
        return None, None


def get_distance_and_duration(lat1, lng1, lat2, lng2, api_key):
    """
    Retourne (distance_km, duration_min) entre deux points
    via l’API Distance Matrix.
    """
    origin = f"{lat1},{lng1}"
    destination = f"{lat2},{lng2}"
    url = (
        "https://maps.googleapis.com/maps/api/distancematrix/json"
        f"?origins={origin}"
        f"&destinations={destination}"
        f"&key={api_key}"
        "&mode=driving"
    )
    resp = requests.get(url).json()
    if resp["status"] == "OK":
        elem = resp["rows"][0]["elements"][0]
        if elem["status"] == "OK":
            dist_m = elem["distance"]["value"]  # mètres
            dur_s = elem["duration"]["value"]  # secondes
            dist_km = round(dist_m / 1000, 2)
            dur_min = round(dur_s / 60, 1)
            return dist_km, dur_min
    return None, None

def get_weather(lat, lng, api_key):
    """
    Récupère la météo actuelle à partir des coordonnées GPS.
    Retourne (température, description) ou (None, None) en cas d'erreur.
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}"
        f"&lon={lng}"
        f"&appid={api_key}"
        f"&units=metric"
        f"&lang=fr"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return temp, description
    else:
        return None, None



# -------------------------------------------------------------------------
#                   INTERFACE UTILISATEUR : ADRESSES
# -------------------------------------------------------------------------
st.header("Adresse de départ")
pickup_input = st.text_input(
    "Tapez votre adresse de départ (puis Entrée)",
    placeholder="Ex: 84 Rue de Sauternes"
)

pickup_suggestions = []
if pickup_input:
    pickup_suggestions = get_address_suggestions(pickup_input, API_KEY)

pickup_options = [s["description"] for s in pickup_suggestions]
selected_pickup = st.selectbox("Suggestions pour l'adresse de départ :", pickup_options)

st.header("Adresse d'arrivée")
destination_input = st.text_input(
    "Tapez votre adresse d'arrivée (puis Entrée)",
    placeholder="Ex: 4949 Métropolitain Est"
)

dest_suggestions = []
if destination_input:
    dest_suggestions = get_address_suggestions(destination_input, API_KEY)

dest_options = [s["description"] for s in dest_suggestions]
selected_destination = st.selectbox("Suggestions pour l'adresse d'arrivée :", dest_options)

# -------------------------------------------------------------------------
#                   BOUTON POUR TOUT CALCULER
# -------------------------------------------------------------------------
if st.button("Prédire le prix"):
    # Vérifier que l'utilisateur a sélectionné une suggestion
    if not selected_pickup or not selected_destination:
        st.error("Veuillez sélectionner une adresse de départ et d'arrivée.")
    else:
        # Récupérer le place_id du départ
        pickup_place_id = None
        for s in pickup_suggestions:
            if s["description"] == selected_pickup:
                pickup_place_id = s["place_id"]
                break

        # Récupérer le place_id de l'arrivée

        dest_place_id = None
        for s in dest_suggestions:
            if s["description"] == selected_destination:
                dest_place_id = s["place_id"]
                break

        if pickup_place_id and dest_place_id:
            # Coords de départ
            pickup_coords = get_place_details(pickup_place_id, API_KEY)
            # Coords d'arrivée
            dest_coords = get_place_details(dest_place_id, API_KEY)

            if pickup_coords and dest_coords:
                dist_km, dur_min = get_distance_and_duration(
                    pickup_coords[0], pickup_coords[1],
                    dest_coords[0], dest_coords[1],
                    API_KEY
                )
                print(pickup_coords)
                if dist_km is not None and dur_min is not None:
                    # Affichage
                    if dur_min > 60:
                        heures = dur_min // 60
                        minutes = dur_min % 60
                        #print(heures, minutes)
                        st.write(f"**Durée estimée** : {int(heures)} heures {int(minutes)} minutes")
                    else:
                        st.write(f"**Distance estimée** : {dist_km} km")

                    # Appel du modèle ML :
                    # => Le modèle attend [distance, durée, prix_km=0.70]
                    # => On lui passe donc ces 3 features

                    donnee_entree = np.array([[dist_km, dur_min, TARIF_PAR_KM]])
                    prediction_modele = modele.predict(donnee_entree)

                    st.write(f"**Prédiction du modèle** : {prediction_modele[0]:.2f} $")
                else:
                    st.error("Impossible de calculer la distance ou la durée.")
            else:
                st.error("Impossible de récupérer les coordonnées GPS.")

            # Affichage de la meteo
            meteo = get_weather(pickup_coords[0], pickup_coords[1], API_METEO)
            if meteo:
                temperature, descrip = meteo
                st.write(f"Temperature : {round(temperature)} degré celsius, principalement : {descrip} en ce moment")


        else:
            st.error("Place ID introuvable pour l'adresse sélectionnée.")
