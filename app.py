# app.py
import os
import streamlit as st
import pandas as pd
import folium
import googlemaps
import polyline
from haversine import haversine
from joblib import load
from dotenv import load_dotenv
from streamlit_folium import st_folium

# Load environment variables
load_dotenv()

# -------------------- CONFIG --------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
MODEL_PATH = r"C:\Users\Zeeshan\Downloads\EV_Route_Optimization_ml\notebooks\models\ev_energy_model_kwh_per_km.pkl"

# -------------------- SAFETY CHECKS --------------------
st.set_page_config(page_title="GROUTE — Chat-driven EV Route Advisor", layout="wide")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_MAPS_API_KEY in environment. Add it to your .env and restart.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at: {MODEL_PATH}\nUpdate MODEL_PATH in the script.")
    st.stop()

# initialize gmaps & model
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

try:
    model = load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

features_needed = list(getattr(model, "feature_names_in_", [])) or [
    'Speed_kmh', 'Distance_Travelled_km', 'Slope_%',
    'Battery_Temperature_C', 'Traffic_Condition', 'Temperature_C', 'Humidity_%'
]

# -------------------- HELPERS --------------------
def geocode_place(place_name):
    try:
        res = gmaps.geocode(place_name)
    except Exception:
        return None
    if not res:
        return None
    loc = res[0]['geometry']['location']
    return (loc['lat'], loc['lng'])

def compute_step_energy(coord_start, coord_end, battery_temp, battery_capacity, curr_soc):
    # approximate haversine distance (km)
    distance_km = haversine(coord_start, coord_end)
    slope = 0.0
    humidity = 50.0
    avg_speed = 30.0

    row = {feat: 0.0 for feat in features_needed}
    if "Speed_kmh" in row: row["Speed_kmh"] = avg_speed
    if "Distance_Travelled_km" in row: row["Distance_Travelled_km"] = distance_km
    if "Slope_%" in row: row["Slope_%"] = slope
    if "Battery_Temperature_C" in row: row["Battery_Temperature_C"] = battery_temp
    if "Humidity_%" in row: row["Humidity_%"] = humidity
    if "Traffic_Condition" in row: row["Traffic_Condition"] = 1.0
    if "Temperature_C" in row: row["Temperature_C"] = 25.0

    df = pd.DataFrame([row])[features_needed]
    pred = model.predict(df)[0]
    # normalize (keep your earlier heuristic)
    kwh_per_km = float(pred) / 10.0
    return kwh_per_km * distance_km

def compute_final_soc(total_energy, battery_capacity, curr_soc):
    available_energy = battery_capacity * curr_soc / 100.0
    extra_needed = max(total_energy - available_energy, 0.0)
    final_soc = max(curr_soc - (total_energy / battery_capacity) * 100.0, 0.0)
    return final_soc, extra_needed

def evaluate_routes(origin_coords, dest_coords, battery_capacity, curr_soc, battery_temp):
    try:
        directions = gmaps.directions(origin_coords, dest_coords,
                                      mode="driving", departure_time="now", alternatives=True)
    except Exception as e:
        raise RuntimeError(f"Google Directions error: {e}")

    route_summaries = []
    for idx, route in enumerate(directions, start=1):
        overview = route.get('overview_polyline', {}).get('points', '')
        coords = polyline.decode(overview) if overview else []
        leg = route.get('legs', [])[0] if route.get('legs') else {}
        distance_km = leg.get('distance', {}).get('value', 0) / 1000.0
        duration_sec = leg.get('duration_in_traffic', {}).get('value') if leg.get('duration_in_traffic') else leg.get('duration', {}).get('value', 0)
        duration_h = duration_sec / 3600.0 if duration_sec else 0.0
        avg_speed = (distance_km / duration_h) if duration_h > 0 else 30.0
        traffic_factor = 30.0 / avg_speed if avg_speed > 0 else 1.0

        total_energy = 0.0
        steps_data = []
        for i in range(len(coords)-1):
            start_pt = coords[i]
            end_pt = coords[i+1]
            step_energy = compute_step_energy(start_pt, end_pt, battery_temp, battery_capacity, curr_soc)
            total_energy += step_energy
            steps_data.append({
                "start": {"lat": start_pt[0], "lng": start_pt[1]},
                "end": {"lat": end_pt[0], "lng": end_pt[1]},
                "distance_km": haversine(start_pt, end_pt),
                "energy_kwh": step_energy
            })

        final_soc, extra_needed = compute_final_soc(total_energy, battery_capacity, curr_soc)

        route_summaries.append({
            "route_index": idx,
            "total_distance_km": distance_km,
            "estimated_energy_kwh": round(total_energy, 2),
            "recommended_speed_kmh": round(avg_speed, 1),
            "extra_energy_needed_kwh": round(extra_needed, 2),
            "final_soc_pct": round(final_soc, 1),
            "traffic_factor": round(traffic_factor, 2),
            "steps": steps_data,
            "duration_sec": duration_sec,
            "coords": coords
        })

    best_route = min(route_summaries, key=lambda x: x['estimated_energy_kwh'], default=None)
    return route_summaries, best_route

def plot_all_routes_map(route_summaries, origin_coords, dest_coords):
    fmap = folium.Map(location=[origin_coords[0], origin_coords[1]], zoom_start=10)
    folium.Marker(location=origin_coords, popup="Start", icon=folium.Icon(color='green')).add_to(fmap)
    folium.Marker(location=dest_coords, popup="Destination", icon=folium.Icon(color='blue')).add_to(fmap)
    colors = ["blue", "red", "purple", "orange", "green"]
    for idx, route in enumerate(route_summaries):
        pts = route.get("coords", [])
        if not pts:
            continue
        folium.PolyLine(pts, color=colors[idx % len(colors)], weight=4, opacity=0.8,
                        tooltip=f"Route {route['route_index']} | {route['total_distance_km']:.1f} km | {route['estimated_energy_kwh']:.1f} kWh").add_to(fmap)
    return fmap

def plot_best_route_map(best_route, origin_coords, dest_coords, battery_capacity, curr_soc):
    fmap = folium.Map(location=[origin_coords[0], origin_coords[1]], zoom_start=11)

    pts = best_route.get("coords", [])
    if pts:
        folium.PolyLine(
            pts, color='blue', weight=6, opacity=0.9,
            tooltip=f"Best Route | {best_route['total_distance_km']:.1f} km"
        ).add_to(fmap)

    # Start & End markers
    folium.Marker(location=origin_coords, popup="Start", icon=folium.Icon(color='green')).add_to(fmap)
    folium.Marker(location=dest_coords, popup="Destination", icon=folium.Icon(color='blue')).add_to(fmap)

    # ------ SOC tracking logic ------
    soc = curr_soc                       # start SOC
    threshold = 15.0                     # show station at 15% SOC
    station_shown = False                # ensure 1 station only
    battery_depletion_point = None       # if SOC reaches 0

    for step in best_route["steps"]:
        distance_km = step["distance_km"]
        used_pct = (step["energy_kwh"] / battery_capacity) * 100.0
        soc -= used_pct

        # battery depletion marker
        if soc <= 0 and battery_depletion_point is None:
            battery_depletion_point = step["end"]

        # show charger when SOC hits 15%
        if soc <= threshold and not station_shown:
            lat = step["end"]["lat"]
            lng = step["end"]["lng"]

            try:
                res = gmaps.places_nearby(
                    location=(lat, lng),
                    radius=4000,
                    type='electric_vehicle_charging_station'
                )
                if res.get("results"):
                    station = res["results"][0]
                    station_loc = station["geometry"]["location"]

                    folium.Marker(
                        location=[station_loc["lat"], station_loc["lng"]],
                        popup=f"Recommended Charging Stop: {station.get('name', 'EV Charger')}",
                        icon=folium.Icon(color='orange', prefix='fa', icon='bolt')
                    ).add_to(fmap)

                else:
                    folium.Marker(
                        location=[lat, lng],
                        popup="No EV station found nearby",
                        icon=folium.Icon(color='gray')
                    ).add_to(fmap)

            except Exception:
                folium.Marker(
                    location=[lat, lng],
                    popup="Charging Station Lookup Failed",
                    icon=folium.Icon(color='gray')
                ).add_to(fmap)

            station_shown = True  # only once

    # battery depletion marker
    if battery_depletion_point:
        folium.Marker(
            location=[battery_depletion_point["lat"], battery_depletion_point["lng"]],
            popup="⚠ Battery will fully deplete here",
            icon=folium.Icon(color='red', icon='exclamation', prefix='fa')
        ).add_to(fmap)

    return fmap


# -------------------- STREAMLIT UI: Chat-driven flow --------------------
# session_state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = "ASK_START"  # ASK_START → ASK_END → ASK_BATTERY → ASK_SOC → ASK_TEMP → DONE
if "route_inputs" not in st.session_state:
    st.session_state.route_inputs = {"start": None, "end": None, "battery_kwh": None, "soc_pct": None, "battery_temp": None}
if "route_results" not in st.session_state:
    st.session_state.route_results = {"all_routes": None, "best_route": None, "origin_coords": None, "dest_coords": None}

# initial assistant prompt
if not st.session_state.chat_history:
    st.session_state.chat_history.append({"role": "assistant", "content": "Hi — I'm the GROUTE assistant. Where are you starting from?"})

# Header and reset
cols = st.columns([0.8, 0.2])
with cols[0]:
    st.title("GROUTE — Chat-driven EV Route Advisor")
with cols[1]:
    if st.button("Reset conversation"):
        # clear all and rerun
        for k in ["chat_history", "chat_state", "route_inputs", "route_results"]:
            if k in st.session_state:
                del st.session_state[k]
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

# Layout: left chat, right maps
col_chat, col_maps = st.columns([1, 1.2])

with col_chat:
    st.header("Chat")
    # show messages using Streamlit chat UI (if available)
    try:
        # Use st.chat_message if available (Streamlit >=1.18+)
        for m in st.session_state.chat_history:
            if m["role"] == "user":
                st.chat_message("user").write(m["content"])
            else:
                st.chat_message("assistant").write(m["content"])
    except Exception:
        # Fallback: simple markdown bubbles
        for m in st.session_state.chat_history:
            if m["role"] == "user":
                st.markdown(f"<div style='background:#d1ffd6;padding:8px;border-radius:8px;margin:6px 0;max-width:85%'><b>You:</b> {m['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#e8f7ff;padding:8px;border-radius:8px;margin:6px 0;max-width:85%'><b>GROUTE:</b> {m['content']}</div>", unsafe_allow_html=True)

    # Chat input pinned at bottom via st.chat_input()
    user_text = None
    try:
        user_text = st.chat_input("Type your reply here")
    except Exception:
        # fallback for older streamlit versions
        user_text = st.text_input("Type your reply here", key="fallback_input")

    if user_text:
        # append user message
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        state = st.session_state.chat_state
        ri = st.session_state.route_inputs

        # FSM transitions
        if state == "ASK_START":
            ri["start"] = user_text.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": "Great — where are you going (destination)?"})
            st.session_state.chat_state = "ASK_END"

        elif state == "ASK_END":
            ri["end"] = user_text.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": "What's your vehicle battery capacity in kWh?"})
            st.session_state.chat_state = "ASK_BATTERY"

        elif state == "ASK_BATTERY":
            try:
                ri["battery_kwh"] = float(user_text.strip())
                st.session_state.chat_history.append({"role": "assistant", "content": "Current State of Charge (%)?"})
                st.session_state.chat_state = "ASK_SOC"
            except ValueError:
                st.session_state.chat_history.append({"role": "assistant", "content": "Please enter a numeric battery capacity (kWh)."})

        elif state == "ASK_SOC":
            try:
                ri["soc_pct"] = float(user_text.strip())
                st.session_state.chat_history.append({"role": "assistant", "content": "What's the battery temperature (°C)?"})
                st.session_state.chat_state = "ASK_TEMP"
            except ValueError:
                st.session_state.chat_history.append({"role": "assistant", "content": "Please enter SOC as a numeric percentage (e.g., 80)."})

        elif state == "ASK_TEMP":
            try:
                ri["battery_temp"] = float(user_text.strip())
                st.session_state.chat_history.append({"role": "assistant", "content": "Thanks — evaluating routes now. This may take a few seconds..."})
                st.session_state.chat_state = "DONE"

                origin_coords = geocode_place(ri["start"])
                dest_coords = geocode_place(ri["end"])
                if origin_coords is None or dest_coords is None:
                    st.session_state.chat_history.append({"role": "assistant", "content": "I couldn't geocode one of the locations. Please check addresses and start over."})
                    st.session_state.chat_state = "ASK_START"
                else:
                    with st.spinner("Fetching routes from Google and evaluating energy..."):
                        try:
                            all_routes, best_route = evaluate_routes(origin_coords, dest_coords,
                                                                     ri["battery_kwh"], ri["soc_pct"], ri["battery_temp"])
                            st.session_state.route_results = {
                                "all_routes": all_routes,
                                "best_route": best_route,
                                "origin_coords": origin_coords,
                                "dest_coords": dest_coords
                            }
                            # assistant reply summary
                            if best_route:
                                summary = (f"I found {len(all_routes)} routes. Most efficient is Route {best_route['route_index']}:\n"
                                           f"- Distance: {best_route['total_distance_km']:.1f} km\n"
                                           f"- Estimated energy: {best_route['estimated_energy_kwh']:.1f} kWh\n"
                                           f"- Final SOC: {best_route['final_soc_pct']:.1f}%\n"
                                           f"- Recommended avg speed (approx): {best_route['recommended_speed_kmh']:.1f} km/h\n"
                                           f"- Traffic factor: {best_route['traffic_factor']:.2f}\n")
                                if best_route['extra_energy_needed_kwh'] > 0:
                                    summary += f"- You'll need ~{best_route['extra_energy_needed_kwh']:.1f} kWh extra — plan to charge en-route.\n"
                                else:
                                    summary += "- No charging required for this route (based on provided SOC & capacity).\n"
                            else:
                                summary = "No routes found."

                            st.session_state.chat_history.append({"role": "assistant", "content": summary})
                        except Exception as e:
                            st.session_state.chat_history.append({"role": "assistant", "content": f"Error while evaluating routes: {e}"})
                            st.session_state.chat_state = "ASK_START"

            except ValueError:
                st.session_state.chat_history.append({"role": "assistant", "content": "Please enter battery temperature as a number (e.g., 25)."})

        elif state == "DONE":
            # After compute: follow-ups
            lower = user_text.strip().lower()
            results = st.session_state.route_results
            best = results.get("best_route")
            if "map" in lower or "show" in lower:
                st.session_state.chat_history.append({"role": "assistant", "content": "Showing maps on the right."})
            elif "speed" in lower:
                if best:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Suggested average speed for best efficiency: ~{best['recommended_speed_kmh']:.1f} km/h."})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "No route computed yet."})
            elif "routes" in lower or "compare" in lower:
                if results.get("all_routes"):
                    st.session_state.chat_history.append({"role": "assistant", "content": "I displayed all routes on the right with their stats."})
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "No routes computed yet."})
            elif "reset" in lower:
                st.session_state.chat_history.append({"role": "assistant", "content": "Resetting conversation."})
                for k in ["chat_history", "chat_state", "route_inputs", "route_results"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
            else:
                st.session_state.chat_history.append({"role": "assistant", "content": "Ask me to 'show maps', 'what speed', or 'compare routes'."})

        # After handling input, re-render to show new chat bubbles and/or maps.
        # Using st.rerun() to clear the chat_input field effect and re-render UI
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

with col_maps:
    st.header("Maps & Route Details")
    results = st.session_state.route_results
    all_routes = results.get("all_routes")
    best_route = results.get("best_route")
    origin_coords = results.get("origin_coords")
    dest_coords = results.get("dest_coords")

    if best_route:
        st.markdown("#### Best route (most energy-efficient)")
        best_map = plot_best_route_map(best_route, origin_coords, dest_coords,
                                       st.session_state.route_inputs["battery_kwh"],
                                       st.session_state.route_inputs["soc_pct"])
        st_folium(best_map, width=700, height=380)

        st.markdown("#### All routes (comparison)")
        all_map = plot_all_routes_map(all_routes, origin_coords, dest_coords)
        st_folium(all_map, width=700, height=380)

        st.markdown("#### Route comparison table")
        df = pd.DataFrame([{
            "Route": r["route_index"],
            "Distance_km": r["total_distance_km"],
            "Time_h": round(r["duration_sec"]/3600.0, 2) if r["duration_sec"] else None,
            "Energy_kWh": r["estimated_energy_kwh"],
            "Final_SOC_%": r["final_soc_pct"],
            "Extra_kWh_needed": r["extra_energy_needed_kwh"],
            "Traffic_factor": r["traffic_factor"]
        } for r in all_routes])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No routes computed yet. Start a chat on the left and provide trip inputs.")

# ------------------ FOOTER ------------------
st.markdown("""
<br><br><hr>
<div style="text-align:center; opacity:0.7; padding:8px; font-size:16px;">
    Made by <b>Zeeshan Khan</b>
</div>
""", unsafe_allow_html=True)
