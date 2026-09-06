import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set
import requests

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
# GESTION DE LA BASE DE DONNÉES LOCALE
# ---------------------------------------------------------------------------
def load_seen_jobs() -> Set[str]:
  """Charge la liste des identifiants d'offres déjà traitées."""
  if os.path.exists(DB_FILE):
    try:
      with open(DB_FILE, "r", encoding="utf-8") as f:
        return set(json.load(f))
    except Exception as e:
      logger.error(f"Erreur lors de la lecture de {DB_FILE} : {e}")
  return set()


def save_seen_jobs(seen_jobs: Set[str]) -> None:
  """Sauvegarde la liste mise à jour dans le fichier JSON."""
  try:
    with open(DB_FILE, "w", encoding="utf-8") as f:
      json.dump(list(seen_jobs), f, ensure_ascii=False, indent=2)
    logger.info(f"Base de données mise à jour ({len(seen_jobs)} offres vues).")
  except Exception as e:
    logger.error(f"Erreur lors de la sauvegarde dans {DB_FILE} : {e}")


# ---------------------------------------------------------------------------
# ANALYSE IA (GEMINI 2.5 FLASH)
# ---------------------------------------------------------------------------
def evaluate_job_with_ai(
    company: str, title: str, location: str, description: str
) -> Dict[str, Any]:
  """Analyse l'offre avec Gemini pour vérifier le profil et le lieu."""
  if not gemini_client:
    logger.warning("Clé API Gemini absente. Validation basique appliquée.")
    return {
        "pertinent": True,
        "score": 5,
        "raison": "Validation par défaut (pas de clé API Gemini).",
        "tags": ["V.I.E"],
    }

  prompt = f"""
Tu es un expert en recrutement d'ingénieurs. Ton rôle est d'analyser une offre d'emploi V.I.E et de déterminer si elle correspond au profil recherché.

--- CRITÈRES STRICTS DU CANDIDAT ---
1. STATUT : V.I.E uniquement.
2. DOMAINES TECHNIQUES RECHERCHÉS (au moins un) :
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
   - Stages, alternances, CDD/CDI classiques hors statut V.I.E.

--- DÉTAILS DE L'OFFRE ---
Entreprise : {company}
Titre du poste : {title}
Localisation : {location}
Description : {description[:3000]}
"""

  try:
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
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
                        "description": "Explication en une phrase simple",
                    },
                    "tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "2 à 4 tags pertinents",
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
  """Envoie une alerte enrichie sur Discord."""
  if not DISCORD_WEBHOOK_URL:
    logger.warning("DISCORD_WEBHOOK_URL non renseigné. Alerte non envoyée.")
    return

  color = 0x3498DB  # Bleu par défaut
  if score >= 8:
    color = 0x2ECC71  # Vert
  elif score >= 6:
    color = 0xF1C40F  # Jaune

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
      "footer": {"text": "Bot Alerte V.I.E • Business France & Corporate ATS"},
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
# SOURCE 1 : BUSINESS FRANCE API
# ---------------------------------------------------------------------------
def fetch_business_france_jobs(seen_jobs: Set[str]) -> None:
  """Interroge la plateforme officielle Business France."""
  api_url = "https://mon-vie-via.businessfrance.fr/api/offers/search"
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      ),
      "Content-Type": "application/json",
  }

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
    logger.info(f"[Business France] Recherche mot-clé : '{query}'")
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

          seen_jobs.add(job_key)

          company = (
              offer.get("companyName")
              or offer.get("organizationName")
              or "Business France"
          )
          title = offer.get("title") or offer.get("jobTitle") or "Poste V.I.E"
          location = (
              offer.get("country")
              or offer.get("cityName")
              or "International"
          )
          desc = (
              offer.get("description")
              or offer.get("summary")
              or f"{title} chez {company}"
          )
          url = f"https://mon-vie-via.businessfrance.fr/offre/{offer_id}"

          analysis = evaluate_job_with_ai(company, title, location, desc)

          if analysis.get("pertinent") and analysis.get("score", 0) >= 6:
            send_discord_alert(
                company=company,
                title=title,
                location=location,
                url=url,
                score=analysis.get("score", 7),
                raison=analysis.get("raison", "Offre pertinente."),
                tags=analysis.get("tags", []),
            )
            time.sleep(1)
    except Exception as e:
      logger.error(f"[Business France] Erreur pour '{query}' : {e}")


# ---------------------------------------------------------------------------
# SOURCE 2 : SITES CARRIÈRES WORKDAY (ALSTOM, AIRBUS, ETC.)
# ---------------------------------------------------------------------------
def check_workday_jobs(
    company_name: str, domain: str, path: str, seen_jobs: Set[str]
) -> None:
  """Interroge directement l'API d'un portail Workday entreprise."""
  api_url = f"https://{domain}.myworkdayjobs.com/wday/cxs/{domain}/{path}/jobs"
  headers = {
      "Content-Type": "application/json",
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      ),
  }
  payload = {
      "appliedFacets": {},
      "limit": 20,
      "offset": 0,
      "searchText": "VIE",
  }

  try:
    logger.info(f"[Workday] Recherche directe chez {company_name}...")
    res = requests.post(api_url, json=payload, headers=headers, timeout=15)
    if res.status_code == 200:
      data = res.json()
      job_postings = data.get("jobPostings", [])

      for job in job_postings:
        bullet_fields = job.get("bulletFields", [])
        ext_id = bullet_fields[0] if bullet_fields else job.get("title")
        job_key = f"WD_{company_name}_{job.get('title')}_{ext_id}"

        if job_key in seen_jobs:
          continue

        seen_jobs.add(job_key)

        title = job.get("title", "")
        location = job.get("locationsText") or job.get(
            "location", "Non renseigné"
        )
        relative_url = job.get("externalPath", "")
        full_url = (
            f"https://{domain}.myworkdayjobs.com/fr-FR/{path}{relative_url}"
        )

        desc = (
            f"Offre publiée directement sur le portail carrières de"
            f" {company_name}.\nTitre : {title}\nLieu : {location}"
        )

        analysis = evaluate_job_with_ai(company_name, title, location, desc)

        if analysis.get("pertinent") and analysis.get("score", 0) >= 6:
          send_discord_alert(
              company=company_name,
              title=f"[Exclusif Site] {title}",
              location=location,
              url=full_url,
              score=analysis.get("score", 7),
              raison=analysis.get("raison", "Publication directe sur le site."),
              tags=analysis.get("tags", ["Direct Site"]),
          )
          time.sleep(1)
  except Exception as e:
    logger.error(f"[Workday] Erreur chez {company_name} : {e}")


# ---------------------------------------------------------------------------
# FONCTION PRINCIPALE
# ---------------------------------------------------------------------------
def main():
  logger.info("=== Démarrage du Bot d'Alerte V.I.E ===")

  # 1. Chargement de l'historique
  seen_jobs = load_seen_jobs()

  # 2. Exécution des scrapers
  fetch_business_france_jobs(seen_jobs)

  # Scrapers directs Workday
  check_workday_jobs(
      company_name="Alstom",
      domain="alstom",
      path="Alstom_Careers",
      seen_jobs=seen_jobs,
  )
  check_workday_jobs(
      company_name="Airbus",
      domain="airbus",
      path="Airbus",
      seen_jobs=seen_jobs,
  )

  # 3. Sauvegarde de l'historique mis à jour
  save_seen_jobs(seen_jobs)
  logger.info("=== Exécution terminée avec succès ===")


if __name__ == "__main__":
  main()
