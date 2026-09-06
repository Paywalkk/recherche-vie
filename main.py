import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set
import requests

# Import de la nouvelle bibliothèque officielle Google GenAI
try:
  from google import genai
  from google.genai import types
except ImportError:
  genai = None

# ---------------------------------------------------------------------------
# CONFIGURATION ET LOGS
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VIE_Bot")

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DB_FILE = "jobs_seen.json"

# Initialisation du client Gemini
gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if (genai and GEMINI_API_KEY)
    else None
)


# ---------------------------------------------------------------------------
# GESTION DE LA BASE DE DONNÉES LOCALE (OBLIGATOIRE POUR ÉVITER LES DOUBLONS)
# ---------------------------------------------------------------------------
def load_seen_jobs() -> Set[str]:
  """Charge la liste des identifiants d'offres déjà traitées."""
  if os.path.exists(DB_FILE):
    try:
      with open(DB_FILE, "r", encoding="utf-8") as f:
        return set(json.load(f))
    except Exception as e:
      logger.error(f"Erreur lors de la lecture de {DB_FILE}: {e}")
  return set()


def save_seen_jobs(seen_jobs: Set[str]) -> None:
  """Sauvegarde la liste mise à jour dans le fichier JSON."""
  try:
    with open(DB_FILE, "w", encoding="utf-8") as f:
      json.dump(list(seen_jobs), f, ensure_ascii=False, indent=2)
    logger.info(
        f"Base de données mise à jour ({len(seen_jobs)} offres enregistrées au"
        " total)."
    )
  except Exception as e:
    logger.error(f"Erreur lors de la sauvegarde dans {DB_FILE}: {e}")


# ---------------------------------------------------------------------------
# ANALYSE DE L'OFFRE VIA GEMINI 2.5 FLASH
# ---------------------------------------------------------------------------
def evaluate_job_with_ai(
    company: str, title: str, location: str, description: str
) -> Dict[str, Any]:
  """Envoie la description de l'offre à l'IA Gemini pour filtrage et scoring."""
  if not gemini_client:
    logger.warning("Clé API Gemini absente. Validation basique appliquée.")
    return {
        "pertinent": True,
        "score": 5,
        "raison": "Validation par défaut (Clé API Gemini non configurée).",
        "tags": ["V.I.E"],
    }

  prompt = f"""
Tu es un expert en recrutement d'ingénieurs. Ton rôle est d'analyser une offre d'emploi V.I.E et de déterminer si elle correspond précisément au profil de l'ingénieur.

--- CRITÈRES STRICTS DU CANDIDAT ---
1. STATUT : V.I.E uniquement.
2. DOMAINES TECHNIQUES RECHERCHÉS (au moins un nécessaire) :
   - Essais & Mise en service (Commissioning)
   - Méthodes / Industrialisation / Procédés
   - Qualité / Assurance Qualité
   - Gestion de projet (PMO) / Direction de chantier / Ingénierie projet
   - Science des Matériaux / Matériaux de structure
   - Énergie / Infrastructures électriques / Mobilité / Ferroviaire
3. LOCALISATION CIBLE :
   - STRICTEMENT HORS EUROPE (Priorités : Afrique, Canada, Amérique latine, Asie).
   - REJETER IMMÉDIATEMENT toute offre située en Europe (France, Allemagne, Espagne, Italie, Belgique, Suisse, Royaume-Uni, etc.).
4. EXCLUSIONS ABSOLUES :
   - Métiers non-ingénieurs : Commerce, Sales, Business Development, Achats, RH, Recrutement, Finance, Comptabilité, Marketing, Support informatique/Helpdesk.
   - Stages, alternances, CDD/CDI classiques hors dispositif V.I.E.

--- DÉTAILS DE L'OFFRE ---
Entreprise : {company}
Titre du poste : {title}
Localisation : {location}
Description : {description[:3000]}
"""

  try:
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "pertinent": {"type": "BOOLEAN"},
                    "score": {
                        "type": "INTEGER",
                        "description": "Note de 1 à 10",
                    },
                    "raison": {
                        "type": "STRING",
                        "description": (
                            "Explication en une phrase concise de la décision"
                        ),
                    },
                    "tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": (
                            "2 à 4 tags pertinents (ex: Essais, Afrique,"
                            " Ferroviaire, Matériaux)"
                        ),
                    },
                },
                "required": ["pertinent", "score", "raison", "tags"],
            },
            temperature=0.1,
        ),
    )
    return json.loads(response.text)
  except Exception as e:
    logger.error(f"Erreur lors de l'analyse Gemini : {e}")
    return {
        "pertinent": False,
        "score": 0,
        "raison": f"Erreur API IA : {e}",
        "tags": [],
    }


# ---------------------------------------------------------------------------
# NOTIFICATION DISCORD
# ---------------------------------------------------------------------------
def send_discord_alert(
    company: str,
    title: str,
    location: str,
    url: str,
    score: int,
    raison: str,
    tags: List[str],
) -> None:
  """Envoie un message enrichi (Embed) sur le salon Discord."""
  if not DISCORD_WEBHOOK_URL:
    logger.warning("DISCORD_WEBHOOK_URL non renseigné. Alerte non envoyée.")
    return

  color = 0x3498DB  # Bleu par défaut
  if score >= 8:
    color = 0x2ECC71  # Vert (Excellente opportunité)
  elif score >= 6:
    color = 0xF1C40F  # Jaune (Bonne opportunité)

  tags_str = (
      " ".join([f"`{t}`" for t in tags]) if tags else "`Ingénieur` `V.I.E`"
  )

  embed = {
      "title": f"🚀 {title}",
      "url": url,
      "color": color,
      "fields": [
          {"name": "🏢 Entreprise", "value": company, "inline": True},
          {"name": "📍 Localisation", "value": location, "inline": True},
          {
              "name": "⭐ Pertinence IA",
              "value": f"**{score}/10**",
              "inline": True,
          },
          {"name": "🏷️ Tags", "value": tags_str, "inline": False},
          {"name": "💡 Analyse IA", "value": raison, "inline": False},
      ],
      "footer": {"text": "Bot Alerte V.I.E • Business France & Grands Groupes"},
      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  }

  payload = {
      "username": "Alerte V.I.E Ingénieur",
      "avatar_url": (
          "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
      ),
      "embeds": [embed],
  }

  try:
    res = requests.post(
        DISCORD_WEBHOOK_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    if res.status_code in [200, 204]:
      logger.info(f"Alerte envoyée sur Discord : {title} ({company})")
    else:
      logger.error(f"Erreur Discord ({res.status_code}) : {res.text}")
  except Exception as e:
    logger.error(f"Erreur d'envoi Discord : {e}")


# ---------------------------------------------------------------------------
# RECUPERATION DES OFFRES DEPUIS L'API BUSINESS FRANCE
# ---------------------------------------------------------------------------
def fetch_business_france_jobs(seen_jobs: Set[str]) -> None:
  """Interroge l'API de Business France pour récupérer les nouvelles offres."""
  api_url = "https://mon-vie-via.businessfrance.fr/api/offers/search"
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      ),
      "Content-Type": "application/json",
  }

  # Mots-clés de recherche couvrant l'ingénierie et les entreprises cibles
  search_queries = [
      "Ingénieur",
      "Essais",
      "Méthodes",
      "Qualité",
      "Projet",
      "Matériaux",
      "Alstom",
      "Schneider",
      "TotalEnergies",
      "Airbus",
      "Safran",
  ]

  for query in search_queries:
    logger.info(f"Recherche Business France avec le mot-clé : '{query}'")
    payload = {"query": query, "limit": 40, "page": 1}

    try:
      response = requests.post(
          api_url, json=payload, headers=headers, timeout=15
      )
      if response.status_code == 200:
        data = response.json()
        items = data.get("items", []) or data.get("offers", [])

        for offer in items:
          offer_id = str(offer.get("id", ""))
          job_key = f"BF_{offer_id}"

          if not offer_id or job_key in seen_jobs:
            continue

          # Marquer l'offre comme vue
          seen_jobs.add(job_key)

          company = (
              offer.get("companyName")
              or offer.get("organizationName")
              or "Entreprise Partenaire"
          )
          title = offer.get("title") or offer.get("jobTitle") or "Poste V.I.E"
          location = (
              offer.get("country")
              or offer.get("cityName")
              or "Destination Internationale"
          )
          desc = (
              offer.get("description")
              or offer.get("summary")
              or f"{title} chez {company}"
          )
          url = f"https://mon-vie-via.businessfrance.fr/offre/{offer_id}"

          # Analyse IA
          analysis = evaluate_job_with_ai(company, title, location, desc)

          if analysis.get("pertinent") and analysis.get("score", 0) >= 6:
            send_discord_alert(
                company=company,
                title=title,
                location=location,
                url=url,
                score=analysis.get("score", 7),
                raison=analysis.get("raison", "Offre correspondant au profil."),
                tags=analysis.get("tags", []),
            )
            time.sleep(1)  # Petite pause pour respecter les limites Discord
      else:
        logger.warning(
            f"API Business France a répondu avec le code {response.status_code}"
        )
    except Exception as e:
      logger.error(
          f"Erreur lors de la requête Business France pour '{query}' : {e}"
      )


# ---------------------------------------------------------------------------
# POINT D'ENTRÉE PRINCIPAL
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  logger.info("=== Démarrage du Bot d'Alerte V.I.E ===")
  seen_jobs = load_seen_jobs()

  # Recherche d'offres
  fetch_business_france_jobs(seen_jobs)

  # Sauvegarde de l'état
  save_seen_jobs(seen_jobs)
  logger.info("=== Exécution terminée avec succès ===")