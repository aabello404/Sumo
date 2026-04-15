"""
WORKFLOW - HOW TO RUN:
(Internal environment file used by train.py and run_with_dashboard.py)
"""
import os
import sys
import uuid
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    raise RuntimeError("Please set SUMO_HOME environment variable")

import traci
from traci.exceptions import TraCIException

SUMO_CFG = "sumo_files/maarif.sumocfg"
MAX_AGENTS = 12

MIN_GREEN = 10
YELLOW_SECS = 3
MAX_GREEN = 60
MAX_WAIT = 300
MAX_QUEUE = 20
MAX_STEPS = 3600

port_counter = 8813
port_lock = threading.Lock()

def get_next_port():
    global port_counter
    with port_lock:
        p = port_counter
        port_counter += 1
        return p

class MultiAgentSUMOEnv:
    def __init__(self, sumo_cfg=SUMO_CFG, max_steps=MAX_STEPS, gui=False):
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.gui = gui
        
        self.port = get_next_port()
        self.label = f"sumo_{uuid.uuid4().hex[:8]}"
        self.conn = None
        self._started = False
        self._step = 0
        # Add this inside MultiAgentSUMOEnv __init__
        self._cached_roads = None
        self.tl_ids = []
        self._phase = {}
        self._phase_start = {}
        self._in_yellow = {}
        self._yellow_start = {}
        
    def _start_sumo(self):
        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [
            binary, "-c", self.sumo_cfg,
            "--no-step-log",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1"
        ]
        traci.start(cmd, port=self.port, label=self.label)
        self.conn = traci.getConnection(self.label)
        self._started = True

    def _init_tl_state(self):
        all_tls = list(self.conn.trafficlight.getIDList())
        self.tl_ids = all_tls[:MAX_AGENTS]
        
        if not self.tl_ids:
            raise RuntimeError("No traffic lights found. Did generate_network.py run properly?")
            
        # Cleaner print statement:
        if hasattr(self, 'target_tl') and self.target_tl:
            print(f"[{self.label}] Agent connected to TL: {self.target_tl}")
        else:
            print(f"[{self.label}] Environment ready. Detected {len(self.tl_ids)} target TLs.")
        all_tls = list(self.conn.trafficlight.getIDList())
        self.tl_ids = all_tls[:MAX_AGENTS]
        if not self.tl_ids:
            raise RuntimeError("No traffic lights found. Did generate_network.py run properly?")
            
        print(f"[{self.label}] Initialised with TLs: {self.tl_ids}")
        
        for tl in self.tl_ids:
            self._phase[tl] = self.conn.trafficlight.getPhase(tl)
            self._phase_start[tl] = 0
            self._in_yellow[tl] = False
            self._yellow_start[tl] = 0

    def _controlled_lanes(self, tl_id):
        lanes = list(set(self.conn.trafficlight.getControlledLanes(tl_id)))
        return lanes[:4]

    def _get_obs(self, tl_id):
        lanes = self._controlled_lanes(tl_id)
        
        queues = []
        waiting_times = []
        
        for lane in lanes:
            queues.append(self.conn.lane.getLastStepHaltingNumber(lane) / MAX_QUEUE)
            waiting_times.append(self.conn.lane.getWaitingTime(lane) / MAX_WAIT)
            
        while len(queues) < 4: queues.append(0.0)
        while len(waiting_times) < 4: waiting_times.append(0.0)
        
        num_phases = len(self.conn.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases)
        phase_norm = self._phase[tl_id] / max(1, num_phases)
        elapsed_norm = (self._step - self._phase_start[tl_id]) / MAX_GREEN
        
        obs = queues[:4] + waiting_times[:4] + [phase_norm, elapsed_norm]
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _get_reward(self, tl_id):
        lanes = self._controlled_lanes(tl_id)
        total_waiting = sum(self.conn.lane.getWaitingTime(l) for l in lanes)
        total_queue = sum(self.conn.lane.getLastStepHaltingNumber(l) for l in lanes)
        return -(total_waiting / 100.0 + total_queue * 0.5)

    def _apply_action(self, tl_id, action):
        if self._in_yellow[tl_id]:
            if self._step - self._yellow_start[tl_id] >= YELLOW_SECS:
                self._in_yellow[tl_id] = False
                num_phases = len(self.conn.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases)
                self._phase[tl_id] = (self._phase[tl_id] + 1) % num_phases
                self._phase_start[tl_id] = self._step
                try:
                    self.conn.trafficlight.setPhase(tl_id, self._phase[tl_id])
                except TraCIException:
                    pass
            return

        elapsed = self._step - self._phase_start[tl_id]
        if action == 1 and elapsed >= MIN_GREEN:
            self._in_yellow[tl_id] = True
            self._yellow_start[tl_id] = self._step
            num_phases = len(self.conn.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases)
            next_phase = (self._phase[tl_id] + 1) % num_phases
            try:
                self.conn.trafficlight.setPhase(tl_id, next_phase)
            except TraCIException:
                pass

    def reset(self):
        if self._started:
            self.conn.close()
            self._started = False
            
        self._step = 0
        self._start_sumo()
        self._init_tl_state()
        
        return {tl: self._get_obs(tl) for tl in self.tl_ids}

    def step(self, actions):
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)
            
        self.conn.simulationStep()
        self._step += 1
        
        obs_dict = {tl: self._get_obs(tl) for tl in self.tl_ids}
        reward_dict = {tl: self._get_reward(tl) for tl in self.tl_ids}
        done_dict = {tl: self._step >= self.max_steps for tl in self.tl_ids}
        info_dict = {tl: {} for tl in self.tl_ids}
        
        return obs_dict, reward_dict, done_dict, info_dict

    def get_dashboard_state(self):
        vehicles = {}
        for vid in self.conn.vehicle.getIDList():
            x, y = self.conn.vehicle.getPosition(vid)
            lon, lat = x, y
            if abs(x) > 180 or abs(y) > 90:
                try:
                    lon, lat = self.conn.simulation.convertGeo(x, y)
                except:
                    pass

            vehicles[vid] = {
                "lat": lat, "lon": lon,
                "speed": self.conn.vehicle.getSpeed(vid),
                "waiting": self.conn.vehicle.getWaitingTime(vid)
            }
            
        tl_states = {}
        for tl in self.tl_ids:
            state_str = self.conn.trafficlight.getRedYellowGreenState(tl)
            tl_states[tl] = {
                "state": state_str,
                "phase": self._phase[tl],
                "lat": 0, "lon": 0 
            }
            lanes = self._controlled_lanes(tl)
            if lanes:
                x, y = self.conn.lane.getShape(lanes[0])[-1]
                lon, lat = x, y
                if abs(x) > 180 or abs(y) > 90:
                    try:
                        lon, lat = self.conn.simulation.convertGeo(x, y)
                    except:
                        pass
                tl_states[tl]["lat"] = lat
                tl_states[tl]["lon"] = lon

        # --- THE PERFORMANCE FIX IS HERE ---
        # Only fetch the roads from SUMO if we haven't done it yet
        if getattr(self, '_cached_roads', None) is None:
            roads = []
            for edge in self.conn.edge.getIDList():
                if not edge.startswith(":"):
                    points = []
                    lane = f"{edge}_0"
                    if lane in self.conn.lane.getIDList():
                        for x, y in self.conn.lane.getShape(lane):
                            lon, lat = x, y
                            if abs(x) > 180 or abs(y) > 90:
                                try:
                                    lon, lat = self.conn.simulation.convertGeo(x, y)
                                except:
                                    pass
                            points.append({"lat": lat, "lon": lon})
                    if points:
                        roads.append({"id": edge, "points": points})
            self._cached_roads = roads

        return {
            "step": self._step,
            "num_vehicles": len(vehicles),
            "vehicles": vehicles,
            "tl_states": tl_states,
            "roads": self._cached_roads # Return the cached version!
        }
        vehicles = {}
        for vid in self.conn.vehicle.getIDList():
            x, y = self.conn.vehicle.getPosition(vid)
            
            # BULLETPROOF CONVERSION: 
            # If x and y are already GPS coordinates, keep them. Otherwise, convert.
            lon, lat = x, y
            if abs(x) > 180 or abs(y) > 90:
                try:
                    lon, lat = self.conn.simulation.convertGeo(x, y)
                except:
                    pass

            vehicles[vid] = {
                "lat": lat, "lon": lon,
                "speed": self.conn.vehicle.getSpeed(vid),
                "waiting": self.conn.vehicle.getWaitingTime(vid)
            }
            
        tl_states = {}
        for tl in self.tl_ids:
            state_str = self.conn.trafficlight.getRedYellowGreenState(tl)
            tl_states[tl] = {
                "state": state_str,
                "phase": self._phase[tl],
                "lat": 0, "lon": 0 
            }
            lanes = self._controlled_lanes(tl)
            if lanes:
                x, y = self.conn.lane.getShape(lanes[0])[-1]
                lon, lat = x, y
                if abs(x) > 180 or abs(y) > 90:
                    try:
                        lon, lat = self.conn.simulation.convertGeo(x, y)
                    except:
                        pass
                tl_states[tl]["lat"] = lat
                tl_states[tl]["lon"] = lon

        roads = []
        for edge in self.conn.edge.getIDList():
            if not edge.startswith(":"):
                points = []
                lane = f"{edge}_0"
                if lane in self.conn.lane.getIDList():
                    for x, y in self.conn.lane.getShape(lane):
                        lon, lat = x, y
                        if abs(x) > 180 or abs(y) > 90:
                            try:
                                lon, lat = self.conn.simulation.convertGeo(x, y)
                            except:
                                pass
                        points.append({"lat": lat, "lon": lon})
                if points:
                    roads.append({"id": edge, "points": points})

        return {
            "step": self._step,
            "num_vehicles": len(vehicles),
            "vehicles": vehicles,
            "tl_states": tl_states,
            "roads": roads
        }
        vehicles = {}
        for vid in self.conn.vehicle.getIDList():
            x, y = self.conn.vehicle.getPosition(vid)
            try:
                lon, lat = self.conn.simulation.convertGeo(x, y)
            except:
                lon, lat = x, y
            vehicles[vid] = {
                "lat": lat, "lon": lon,
                "speed": self.conn.vehicle.getSpeed(vid),
                "waiting": self.conn.vehicle.getWaitingTime(vid)
            }
            
        tl_states = {}
        for tl in self.tl_ids:
            state_str = self.conn.trafficlight.getRedYellowGreenState(tl)
            tl_states[tl] = {
                "state": state_str,
                "phase": self._phase[tl],
                "lat": 0, "lon": 0 
            }
            lanes = self._controlled_lanes(tl)
            if lanes:
                x, y = self.conn.lane.getShape(lanes[0])[-1]
                try:
                    lon, lat = self.conn.simulation.convertGeo(x, y)
                    tl_states[tl]["lat"] = lat
                    tl_states[tl]["lon"] = lon
                except:
                    pass

        roads = []
        for edge in self.conn.edge.getIDList():
            if not edge.startswith(":"):
                points = []
                lane = f"{edge}_0"
                if lane in self.conn.lane.getIDList():
                    for x, y in self.conn.lane.getShape(lane):
                        try:
                            lon, lat = self.conn.simulation.convertGeo(x, y)
                            points.append({"lat": lat, "lon": lon})
                        except:
                            pass
                if points:
                    roads.append({"id": edge, "points": points})

        return {
            "step": self._step,
            "num_vehicles": len(vehicles),
            "vehicles": vehicles,
            "tl_states": tl_states,
            "roads": roads
        }

    def close(self):
        if self._started:
            self.conn.close()
            self._started = False


class SingleAgentWrapper(gym.Env):
    def __init__(self, tl_index, sumo_cfg=SUMO_CFG, max_steps=MAX_STEPS, gui=False):
        super().__init__()
        self.tl_index = tl_index
        self.env = MultiAgentSUMOEnv(sumo_cfg, max_steps, gui)
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.target_tl = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_dict = self.env.reset()
        if self.tl_index >= len(self.env.tl_ids):
            self.tl_index = len(self.env.tl_ids) - 1
        self.target_tl = self.env.tl_ids[self.tl_index]
        return obs_dict[self.target_tl], {}

    def step(self, action):
        actions = {tl: 0 for tl in self.env.tl_ids}
        actions[self.target_tl] = action
        
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(actions)
        
        return (
            obs_dict[self.target_tl],
            reward_dict[self.target_tl],
            done_dict[self.target_tl],
            False,
            info_dict[self.target_tl]
        )

    def close(self):
        self.env.close()

def detect_num_tls(sumo_cfg=SUMO_CFG):
    port = get_next_port()
    label = f"detect_{uuid.uuid4().hex[:8]}"
    cmd = ["sumo", "-c", sumo_cfg, "--no-step-log"]
    traci.start(cmd, port=port, label=label)
    conn = traci.getConnection(label)
    num = len(conn.trafficlight.getIDList())
    conn.close()
    return min(num, MAX_AGENTS)