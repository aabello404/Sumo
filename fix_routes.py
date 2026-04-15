import os
import sys
import subprocess

# Auto-detect SUMO_HOME or use default Windows path
sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
random_trips_path = os.path.join(sumo_home, "tools", "randomTrips.py")

net_file = "sumo_files/maarif.net.xml"
trips_file = "sumo_files/maarif.trips.xml"
rou_file = "sumo_files/maarif.rou.xml"

print("Regenerating traffic routes for your custom map...")

# Run SUMO's random trip generator with the exact same density as before
cmd = [
    sys.executable, random_trips_path,
    "-n", net_file,
    "-b", "0", "-e", "3600",
    "-p", "2.0",
    "--seed", "42",
    "--min-distance", "80",
    "--fringe-factor", "3",
    "--validate", # This flag forces SUMO to check the new map for valid paths!
    "-o", trips_file,
    "-r", rou_file
]

try:
    subprocess.run(cmd, check=True)
    print("\n✅ SUCCESS: New routes generated!")
    print("🚗 You can now run: python run_with_dashboard.py")
except subprocess.CalledProcessError:
    print("\n❌ ERROR: Failed to generate routes.")