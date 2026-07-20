import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from collections import defaultdict
from folium.plugins import MarkerCluster
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np
import os
from google import genai
from google.genai import types
from openai import OpenAI

st.set_page_config(layout="wide")
MODEL_NAME = "gemini-2.5-flash-lite"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

gemini_client = OpenAI(
    base_url=GEMINI_BASE_URL,
    api_key=st.secrets["GOOGLE_API_KEY"]
)

# Load your enriched events (replace with your actual loader)
def load_events():
    # Example: from JSON file
    import json
    with open("final_events.json", "r", encoding="utf-8") as f:
        return json.load(f)

events = load_events()

# -------------------------------
# MAP SECTION
# -------------------------------
st.title("India Logistical Issue Locations Dashboard")

rows = []
for e in events:
    for loc in e["location_details"]:
        rows.append({
            "lat": loc["lat"],
            "lon": loc["lon"],
            "city": loc["city"],
            "state": loc["state"],
            "title": e["title"],
            "link": e["link"]
        })
df = pd.DataFrame(rows)

# Create Folium map
m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

# Add marker clustering
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    folium.Marker(
        [row["lat"], row["lon"]],
        popup=f"{row['title']} ({row['city']}, {row['state']})\n{row['link']}",
        tooltip=row["city"],
        icon=folium.Icon(color="blue", icon="info-sign")  # ✅ fixes broken image icons
    ).add_to(marker_cluster)

with st.container():
    st_folium(m, width=1600, height=800)

# Load FAISS index and mapping
index = faiss.read_index("events.index")
with open("id_to_event.json", "r", encoding="utf-8") as f:
    id_to_event = json.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# CHAT SECTION
# -------------------------------
st.subheader("Ask about events in a state or city")

user_query = st.text_input("Enter your question (e.g., 'events in Delhi')")

if st.button("Summarize"):
    # Embed the query
    q_emb = model.encode([user_query])
    D, I = index.search(np.array(q_emb), k=len(id_to_event))  # search across all

    # Filter by similarity score
    threshold = 1  # tune this value
    matched_events = [id_to_event[str(i)] for idx, i in enumerate(I[0]) if D[0][idx] < threshold]
    #matched_events = []
    #for idx, i in enumerate(I[0]):
    #    if D[0][idx] < 1.0:  # adjust threshold
    #        matched_events.append(id_to_event[str(i)])

    if not matched_events:
        st.write("No events found for that location.")
    else:
        # ✅ Show all matched events first
        st.subheader("Matched events for this location")
        for e in matched_events:
            st.write(
                f"- {e['title']} "
                f"(Reporting City: {e.get('reporting_city', '')}, "
                f"Reporting State: {e.get('reporting_state', '')}, "
                f"Event Dates: {', '.join(e.get('event_dates', []))}, "
                f"Reporting Date: {e.get('reporting_date', '')})"
            )
        # Build prompt with system role
        st.write('---')
        st.subheader("My Opinion based on the Searched Results")
        
        system_prompt = (
            "You are a logistics support agent. "
            "You have information on events that may cause road blocks, diversions, or transport problems for company trucks. "
            "From the provided events, recommend only those that are relevant for logistics planning."
        )

        # 1. Cleanly build the text block using a list to avoid serialization glitches
        prompt_lines = [f"Here are events related to {user_query}:\n"]
        for e in matched_events:
            event_dates = e.get('event_dates')
            date_str = ", ".join(event_dates) if isinstance(event_dates, list) else str(event_dates or "")
            
            line = (
                f"- {e.get('title', 'Unknown Event')} "
                f"(Reporting City: {e.get('reporting_city', '')}, "
                f"Reporting State: {e.get('reporting_state', '')}, "
                f"Event Dates: {date_str}, "
                f"Reporting Date: {e.get('reporting_date', '')})"
            )
            prompt_lines.append(line)

        # Combine everything into a single clean string
        user_prompt = "\n".join(prompt_lines)

        # 2. Call Gemini ensuring values are clean strings
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call Gemini using the OpenAI Client structure with error handling retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = gemini_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    # tools=tools, # Uncomment this only if 'tools' is defined elsewhere in your script
                )
                
                # Print the response text to the dashboard
                st.write(response.choices[0].message.content)
                break  # Success, exit retry loop
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to generate output after {max_retries} attempts: {e}")
                else:
                    continue  # Try again
