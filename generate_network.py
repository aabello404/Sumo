"""
WORKFLOW - HOW TO RUN:
Step 1: pip install -r requirements.txt
Step 2: $env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
Step 3: python generate_network.py
"""
import os
import sys
import subprocess
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

def main():
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        print("ERROR: SUMO_HOME environment variable not set.")
        sys.exit(1)

    os.makedirs("sumo_files", exist_ok=True)
    
    osm_file = "sumo_files/maarif.osm"
    net_file = "sumo_files/maarif.net.xml"
    trips_file = "sumo_files/maarif.trips.xml"
    rou_file = "sumo_files/maarif.rou.xml"
    cfg_file = "sumo_files/maarif.sumocfg"

    # STEP 1 - Download OSM data
    print("\n--- STEP 1: Downloading OSM Data ---")
    bbox = "33.5700,-7.6450,33.5900,-7.6200" # Wider area of Maarif # south, west, north, east
    query = f"""
    [out:xml][timeout:25];
    (
      way["highway"]({bbox});
      node["traffic_signals"]({bbox});
    );
    (._;>;);
    out body;
    """
    
    download_needed = True
    if os.path.exists(osm_file) and os.path.getsize(osm_file) > 5120:
        print(f"File {osm_file} already exists and is >5KB. Skipping download.")
        download_needed = False

    if download_needed:
        apis = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"
        ]
        success = False
        for api in apis:
            try:
                print(f"Trying Overpass API: {api}")
                req = urllib.request.Request(api, data=query.encode('utf-8'))
                with urllib.request.urlopen(req) as response, open(osm_file, 'wb') as f:
                    f.write(response.read())
                success = True
                print("Download successful.")
                break
            except urllib.error.URLError as e:
                print(f"Failed to fetch from {api}: {e}")
        
        if not success:
            print("ERROR: Failed to download OSM data.")
            sys.exit(1)

    # STEP 2 - Convert OSM to SUMO network
    print("\n--- STEP 2: Converting OSM to SUMO Network ---")
    netconvert_cmd = [
        "netconvert",
        "--osm-files", osm_file,
        "--output-file", net_file,
        "--geometry.remove",
        "--roundabouts.guess",
        "--junctions.join",
        "--tls.guess", "true",        # Added this to force guessing
        "--tls.guess-signals", "true",
        "--tls.join",
        "--tls.default-type", "actuated",
        "--keep-edges.by-vclass", "passenger",
        "--remove-edges.isolated",
        #"--proj.utm",
        "--no-warnings"
    ]
    
    typemap_path = os.path.join(sumo_home, "data", "typemap", "osmNetconvert.typ.xml")
    if os.path.exists(typemap_path):
        netconvert_cmd.extend(["--type-files", typemap_path])
        
    subprocess.run(netconvert_cmd, check=True)

    # STEP 3 - Generate random vehicle routes
    print("\n--- STEP 3: Generating Random Routes ---")
    random_trips_path = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.exists(random_trips_path):
        random_trips_path = os.path.join(sumo_home, "tools", "trip", "randomTrips.py")
        
    trips_cmd = [
        sys.executable, random_trips_path,
        "-n", net_file,
        "-b", "0", "-e", "3600",
        "-p", "2.0",
        "--seed", "42",
        "--min-distance", "80",
        "--fringe-factor", "3",
        "--validate",
        "-o", trips_file,
        "-r", rou_file
    ]
    subprocess.run(trips_cmd, check=True)

    # STEP 4 - Write sumocfg
    print("\n--- STEP 4: Writing sumocfg ---")
    cfg_content = f"""<configuration>
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{os.path.basename(rou_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>"""
    with open(cfg_file, "w") as f:
        f.write(cfg_content)

    print("\n--- Network Parsing ---")
    tree = ET.parse(net_file)
    root = tree.getroot()
    tl_logic_elements = root.findall("tlLogic")
    
    tl_ids = list(set([tl.get("id") for tl in tl_logic_elements]))
    print(f"Found {len(tl_ids)} traffic lights:")
    for tid in tl_ids:
        print(f"  - {tid}")
        
    print(f"\ntrain.py will spawn {len(tl_ids)} agents automatically.")

if __name__ == "__main__":
    main()