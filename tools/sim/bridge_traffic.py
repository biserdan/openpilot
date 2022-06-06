#!/usr/bin/env python3
import argparse
import logging
import math
import os
import signal
import threading
import time
from multiprocessing import Process, Queue
from typing import Any

import carla  # pylint: disable=import-error
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from numpy import random

import cereal.messaging as messaging
from cereal import log
from cereal.visionipc.visionipc_pyx import VisionIpcServer, VisionStreamType  # pylint: disable=no-name-in-module, import-error
from common.basedir import BASEDIR
from common.numpy_fast import clip
from common.params import Params
from common.realtime import DT_DMON, Ratekeeper
from selfdrive.car.honda.values import CruiseButtons
from selfdrive.test.helpers import set_params_enabled
from tools.sim.lib.can import can_function

W, H = 1928, 1208
REPEAT_COUNTER = 5
PRINT_DECIMATION = 100
STEER_RATIO = 15.

pm = messaging.PubMaster(['roadCameraState', 'wideRoadCameraState', 'sensorEvents', 'can', "gpsLocationExternal"])
sm = messaging.SubMaster(['carControl', 'controlsState'])

def parse_args(add_args=None):
  parser = argparse.ArgumentParser(description='Bridge between CARLA and openpilot.')
  parser.add_argument('--joystick', action='store_true')
  parser.add_argument('--high_quality', action='store_true')
  parser.add_argument('--town', type=str, default='Town04_Opt')
  parser.add_argument('--spawn_point', dest='num_selected_spawn_point', type=int, default=16)
  # add arguments for traffic manager
  parser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port to communicate with TM (default: 8000)')
  parser.add_argument('-n', '--number-of-vehicles', metavar='N', default=10, type=int, help='Number of vehicles (default: 30)')
  parser.add_argument('--filterv', metavar='PATTERN', default='vehicle.*', help='Filter vehicle model (default: "vehicle.*")')
  parser.add_argument('--generationv', metavar='G', default='All', help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
  parser.add_argument('--respawn', action='store_true', default=False, help='Automatically respawn dormant vehicles (only in large maps)')
  parser.add_argument('--hybrid', action='store_true', help='Activate hybrid mode for Traffic Manager')
  parser.add_argument('-s', '--seed', metavar='S', type=int, help='Set random device seed and deterministic mode for Traffic Manager')
  parser.add_argument('--car-lights-on', action='store_true', default=False, help='Enable automatic car light management')
  parser.add_argument('--safe', action='store_true', default=True, help='Avoid spawning vehicles prone to accidents')
  parser.add_argument('--hero', action='store_true', default=True, help='Set one of the vehicles as hero')

  return parser.parse_args(add_args)


class VehicleState:
  def __init__(self):
    self.speed = 0.0
    self.angle = 0.0
    self.bearing_deg = 0.0
    self.vel = carla.Vector3D()
    self.cruise_button = 0
    self.is_engaged = False
    self.ignition = True


def steer_rate_limit(old, new):
  # Rate limiting to 0.5 degrees per step
  limit = 0.5
  if new > old + limit:
    return old + limit
  elif new < old - limit:
    return old - limit
  else:
    return new


class Camerad:
  def __init__(self):
    self.frame_road_id = 0
    self.frame_wide_id = 0
    self.vipc_server = VisionIpcServer("camerad")

    # TODO: remove RGB buffers once the last RGB vipc subscriber is removed
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_RGB_ROAD, 4, True, W, H)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_ROAD, 5, False, W, H)

    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_RGB_WIDE_ROAD, 4, True, W, H)
    self.vipc_server.create_buffers(VisionStreamType.VISION_STREAM_WIDE_ROAD, 5, False, W, H)
    self.vipc_server.start_listener()

    # set up for pyopencl rgb to yuv conversion
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue(self.ctx)
    cl_arg = f" -DHEIGHT={H} -DWIDTH={W} -DRGB_STRIDE={W * 3} -DUV_WIDTH={W // 2} -DUV_HEIGHT={H // 2} -DRGB_SIZE={W * H} -DCL_DEBUG "

    # TODO: move rgb_to_yuv.cl to local dir once the frame stream camera is removed
    kernel_fn = os.path.join(BASEDIR, "selfdrive", "camerad", "transforms", "rgb_to_yuv.cl")
    with open(kernel_fn) as f:
      prg = cl.Program(self.ctx, f.read()).build(cl_arg)
      self.krnl = prg.rgb_to_yuv
    self.Wdiv4 = W // 4 if (W % 4 == 0) else (W + (4 - W % 4)) // 4
    self.Hdiv4 = H // 4 if (H % 4 == 0) else (H + (4 - H % 4)) // 4

  def cam_callback_road(self, image):
    self._cam_callback(image, self.frame_road_id, 'roadCameraState',
                       VisionStreamType.VISION_STREAM_RGB_ROAD, VisionStreamType.VISION_STREAM_ROAD)
    self.frame_road_id += 1

  def cam_callback_wide_road(self, image):
    self._cam_callback(image, self.frame_wide_id, 'wideRoadCameraState',
                       VisionStreamType.VISION_STREAM_RGB_WIDE_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD)
    self.frame_wide_id += 1

  def _cam_callback(self, image, frame_id, pub_type, rgb_type, yuv_type):
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (H, W, 4))
    img = img[:, :, [0, 1, 2]].copy()

    # convert RGB frame to YUV
    rgb = np.reshape(img, (H, W * 3))
    rgb_cl = cl_array.to_device(self.queue, rgb)
    yuv_cl = cl_array.empty_like(rgb_cl)
    self.krnl(self.queue, (np.int32(self.Wdiv4), np.int32(self.Hdiv4)), None, rgb_cl.data, yuv_cl.data).wait()
    yuv = np.resize(yuv_cl.get(), rgb.size // 2)
    eof = int(frame_id * 0.05 * 1e9)

    # TODO: remove RGB send once the last RGB vipc subscriber is removed
    self.vipc_server.send(rgb_type, img.tobytes(), frame_id, eof, eof)
    self.vipc_server.send(yuv_type, yuv.data.tobytes(), frame_id, eof, eof)

    dat = messaging.new_message(pub_type)
    msg = {
      "frameId": image.frame,
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0]
    }
    setattr(dat, pub_type, msg)
    pm.send(pub_type, dat)

def imu_callback(imu, vehicle_state):
  vehicle_state.bearing_deg = math.degrees(imu.compass)
  dat = messaging.new_message('sensorEvents', 2)
  dat.sensorEvents[0].sensor = 4
  dat.sensorEvents[0].type = 0x10
  dat.sensorEvents[0].init('acceleration')
  dat.sensorEvents[0].acceleration.v = [imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z]
  # copied these numbers from locationd
  dat.sensorEvents[1].sensor = 5
  dat.sensorEvents[1].type = 0x10
  dat.sensorEvents[1].init('gyroUncalibrated')
  dat.sensorEvents[1].gyroUncalibrated.v = [imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z]
  pm.send('sensorEvents', dat)


def panda_state_function(vs: VehicleState, exit_event: threading.Event):
  pm = messaging.PubMaster(['pandaStates'])
  while not exit_event.is_set():
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': vs.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec'
    }
    pm.send('pandaStates', dat)
    time.sleep(0.5)


def peripheral_state_function(exit_event: threading.Event):
  pm = messaging.PubMaster(['peripheralState'])
  while not exit_event.is_set():
    dat = messaging.new_message('peripheralState')
    dat.valid = True
    # fake peripheral state data
    dat.peripheralState = {
      'pandaType': log.PandaState.PandaType.blackPanda,
      'voltage': 12000,
      'current': 5678,
      'fanSpeedRpm': 1000
    }
    pm.send('peripheralState', dat)
    time.sleep(0.5)


def gps_callback(gps, vehicle_state):
  dat = messaging.new_message('gpsLocationExternal')

  # transform vel from carla to NED
  # north is -Y in CARLA
  velNED = [
    -vehicle_state.vel.y,  # north/south component of NED is negative when moving south
    vehicle_state.vel.x,  # positive when moving east, which is x in carla
    vehicle_state.vel.z,
  ]

  dat.gpsLocationExternal = {
    "timestamp": int(time.time() * 1000),
    "flags": 1,  # valid fix
    "accuracy": 1.0,
    "verticalAccuracy": 1.0,
    "speedAccuracy": 0.1,
    "bearingAccuracyDeg": 0.1,
    "vNED": velNED,
    "bearingDeg": vehicle_state.bearing_deg,
    "latitude": gps.latitude,
    "longitude": gps.longitude,
    "altitude": gps.altitude,
    "speed": vehicle_state.speed,
    "source": log.GpsLocationData.SensorSource.ublox,
  }

  pm.send('gpsLocationExternal', dat)


def fake_driver_monitoring(exit_event: threading.Event):
  pm = messaging.PubMaster(['driverState', 'driverMonitoringState'])
  while not exit_event.is_set():
    # dmonitoringmodeld output
    dat = messaging.new_message('driverState')
    dat.driverState.faceProb = 1.0
    pm.send('driverState', dat)

    # dmonitoringd output
    dat = messaging.new_message('driverMonitoringState')
    dat.driverMonitoringState = {
      "faceDetected": True,
      "isDistracted": False,
      "awarenessStatus": 1.,
    }
    pm.send('driverMonitoringState', dat)

    time.sleep(DT_DMON)


def can_function_runner(vs: VehicleState, exit_event: threading.Event):
  i = 0
  while not exit_event.is_set():
    can_function(pm, vs.speed, vs.angle, i, vs.cruise_button, vs.is_engaged)
    time.sleep(0.01)
    i += 1


def connect_carla_client():
  client = carla.Client("127.0.0.1", 2000)
  client.set_timeout(5)
  return client


class CarlaBridge:

  def __init__(self, arguments):
    set_params_enabled()

    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    Params().put("CalibrationParams", msg.to_bytes())

    self._args = arguments
    self._carla_objects = []
    self._vehicles_list = []
    self._camerad = None
    self._exit_event = threading.Event()
    self._threads = []
    self._keep_alive = True
    self.started = False
    signal.signal(signal.SIGTERM, self._on_shutdown)
    self._exit = threading.Event()

  def _on_shutdown(self, signal, frame):
    self._keep_alive = False

  def bridge_keep_alive(self, q: Queue, retries: int):
    try:
      while self._keep_alive:
        try:
          self._run(q)
          break
        except RuntimeError as e:
          self.close()
          if retries == 0:
            raise

          # Reset for another try
          self._carla_objects = []
          self._vehicles_list = []
          self._threads = []
          self._exit_event = threading.Event()

          retries -= 1
          if retries <= -1:
            print(f"Restarting bridge. Error: {e} ")
          else:
            print(f"Restarting bridge. Retries left {retries}. Error: {e} ")
    finally:
      # Clean up resources in the opposite order they were created.
      self.close()

  def get_actor_blueprints(self, world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

  def _run(self, q: Queue):
    client = connect_carla_client()
    world = client.load_world(self._args.town)

    settings = world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearSunset)

    if not self._args.high_quality:
      world.unload_map_layer(carla.MapLayer.Foliage)
      world.unload_map_layer(carla.MapLayer.Buildings)
      world.unload_map_layer(carla.MapLayer.ParkedVehicles)
      world.unload_map_layer(carla.MapLayer.Props)
      world.unload_map_layer(carla.MapLayer.StreetLights)
      world.unload_map_layer(carla.MapLayer.Particles)

    blueprint_library = world.get_blueprint_library()

    world_map = world.get_map()

    # BA22
    # creates TM instance by a carla client using tm_port to connect
    traffic_manager = client.get_trafficmanager(self._args.tm_port)
    # vehicle get spawned around hero
    if self._args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
    # physics are activated when in radius of hero
    if self._args.hybrid:
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
    # make simulation deterministic
    if self._args.seed is not None:
        traffic_manager.set_random_device_seed(self._args.seed)

    # get blueprints of vehicles with specification from arg filterv and generationv
    blueprints = self.get_actor_blueprints(world, self._args.filterv, self._args.generationv)


    # disable bicycles in simulation because openpilot don't care about them
    if self._args.safe:
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    
    # sorts the blueprints based on their id
    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    # gets spawn_points on specified map
    spawn_points = world_map.get_spawn_points()
    # sets the number of possible spawn_points
    number_of_spawn_points = len(spawn_points)

    # shuffles the spawn_points randomly if number of spawn_points is bigger than number_of_vehicles
    if self._args.number_of_vehicles < number_of_spawn_points:
        print('Number of spawn points: ', number_of_spawn_points)
        print('Number of vehicels: ', self._args.number_of_vehicles)
        random.shuffle(spawn_points)
    # changes vehicles number to number of spawn points since the number of vehicles exceed spawn points
    elif self._args.number_of_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, self._args.number_of_vehicles, number_of_spawn_points)
        self._args.number_of_vehicles = number_of_spawn_points

    # preparation for spawning vehicles with an autopilot
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    synchronous_master = True

    # prepare all vehicles spawning in a list
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= args.number_of_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(
            blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
            blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        else:
            blueprint.set_attribute('role_name', 'autopilot')

        # append the cars to list and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
    # spawn cars with applied settings
    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            self._vehicles_list.append(response.actor_id)

    print('spawned %d vehicles' % (len(self._vehicles_list)))

    # set automatic vehicle lights update if specified
    if args.car_lights_on:
        all_vehicle_actors = world.get_actors(self._vehicles_list)
        for actor in all_vehicle_actors:
            traffic_manager.update_vehicle_lights(actor, True)




    # openpilot
    blueprint_op = blueprint_library.filter('vehicle.tesla.*')[1]
    blueprint_op.set_attribute('role_name', 'hero')
    vehicle_bp = blueprint_op
    spawn_points = world_map.get_spawn_points()
    
    # default 16
    assert len(spawn_points) > self._args.num_selected_spawn_point, f'''No spawn point {self._args.num_selected_spawn_point}, try a value between 0 and
      {len(spawn_points)} for this town.'''
    spawn_point = spawn_points[self._args.num_selected_spawn_point]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    self._carla_objects.append(vehicle)
    max_steer_angle = vehicle.get_physics_control().wheels[0].max_steer_angle

    # make tires less slippery
    # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
    physics_control = vehicle.get_physics_control()
    physics_control.mass = 2326
    # physics_control.wheels = [wheel_control]*4
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    vehicle.apply_physics_control(physics_control)

    transform = carla.Transform(carla.Location(x=0.8, z=1.13))

    def create_camera(fov, callback):
      blueprint = blueprint_library.find('sensor.camera.rgb')
      blueprint.set_attribute('image_size_x', str(W))
      blueprint.set_attribute('image_size_y', str(H))
      blueprint.set_attribute('fov', str(fov))
      if not self._args.high_quality:
        blueprint.set_attribute('enable_postprocess_effects', 'False')
      camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
      camera.listen(callback)
      return camera

    self._camerad = Camerad()
    road_camera = create_camera(fov=40, callback=self._camerad.cam_callback_road)
    road_wide_camera = create_camera(fov=120, callback=self._camerad.cam_callback_wide_road)  # fov bigger than 120 shows unwanted artifacts

    self._carla_objects.extend([road_camera, road_wide_camera])

    vehicle_state = VehicleState()

    # reenable IMU
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    imu.listen(lambda imu: imu_callback(imu, vehicle_state))

    gps_bp = blueprint_library.find('sensor.other.gnss')
    gps = world.spawn_actor(gps_bp, transform, attach_to=vehicle)
    gps.listen(lambda gps: gps_callback(gps, vehicle_state))

    self._carla_objects.extend([imu, gps])
    # launch fake car threads
    self._threads.append(threading.Thread(target=panda_state_function, args=(vehicle_state, self._exit_event,)))
    self._threads.append(threading.Thread(target=peripheral_state_function, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=fake_driver_monitoring, args=(self._exit_event,)))
    self._threads.append(threading.Thread(target=can_function_runner, args=(vehicle_state, self._exit_event,)))
    for t in self._threads:
      t.start()

    # init
    throttle_ease_out_counter = REPEAT_COUNTER
    brake_ease_out_counter = REPEAT_COUNTER
    steer_ease_out_counter = REPEAT_COUNTER

    vc = carla.VehicleControl(throttle=0, steer=0, brake=0, reverse=False)

    is_openpilot_engaged = False
    throttle_out = steer_out = brake_out = 0.
    throttle_op = steer_op = brake_op = 0.
    throttle_manual = steer_manual = brake_manual = 0.

    old_steer = old_brake = old_throttle = 0.
    throttle_manual_multiplier = 0.7  # keyboard signal is always 1
    brake_manual_multiplier = 0.7  # keyboard signal is always 1
    steer_manual_multiplier = 45 * STEER_RATIO  # keyboard signal is always 1

    # Simulation tends to be slow in the initial steps. This prevents lagging later
    for _ in range(20):
      world.tick()

    # loop
    rk = Ratekeeper(100, print_delay_threshold=0.05)

    while self._keep_alive:
      # 1. Read the throttle, steer and brake from op or manual controls
      # 2. Set instructions in Carla
      # 3. Send current carstate to op via can

      cruise_button = 0
      throttle_out = steer_out = brake_out = 0.0
      throttle_op = steer_op = brake_op = 0.0
      throttle_manual = steer_manual = brake_manual = 0.0

      # --------------Step 1-------------------------------
      if not q.empty():
        message = q.get()
        m = message.split('_')
        if m[0] == "steer":
          steer_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "throttle":
          throttle_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "brake":
          brake_manual = float(m[1])
          is_openpilot_engaged = False
        elif m[0] == "reverse":
          cruise_button = CruiseButtons.CANCEL
          is_openpilot_engaged = False
        elif m[0] == "cruise":
          if m[1] == "down":
            cruise_button = CruiseButtons.DECEL_SET
            is_openpilot_engaged = True
          elif m[1] == "up":
            cruise_button = CruiseButtons.RES_ACCEL
            is_openpilot_engaged = True
          elif m[1] == "cancel":
            cruise_button = CruiseButtons.CANCEL
            is_openpilot_engaged = False
        elif m[0] == "ignition":
          vehicle_state.ignition = not vehicle_state.ignition
        elif m[0] == "quit":
          break

        throttle_out = throttle_manual * throttle_manual_multiplier
        steer_out = steer_manual * steer_manual_multiplier
        brake_out = brake_manual * brake_manual_multiplier

        old_steer = steer_out
        old_throttle = throttle_out
        old_brake = brake_out

      if is_openpilot_engaged:
        sm.update(0)

        # TODO gas and brake is deprecated
        throttle_op = clip(sm['carControl'].actuators.accel / 1.6, 0.0, 1.0)
        brake_op = clip(-sm['carControl'].actuators.accel / 4.0, 0.0, 1.0)
        steer_op = sm['carControl'].actuators.steeringAngleDeg

        throttle_out = throttle_op
        steer_out = steer_op
        brake_out = brake_op

        steer_out = steer_rate_limit(old_steer, steer_out)
        old_steer = steer_out

      else:
        if throttle_out == 0 and old_throttle > 0:
          if throttle_ease_out_counter > 0:
            throttle_out = old_throttle
            throttle_ease_out_counter += -1
          else:
            throttle_ease_out_counter = REPEAT_COUNTER
            old_throttle = 0

        if brake_out == 0 and old_brake > 0:
          if brake_ease_out_counter > 0:
            brake_out = old_brake
            brake_ease_out_counter += -1
          else:
            brake_ease_out_counter = REPEAT_COUNTER
            old_brake = 0

        if steer_out == 0 and old_steer != 0:
          if steer_ease_out_counter > 0:
            steer_out = old_steer
            steer_ease_out_counter += -1
          else:
            steer_ease_out_counter = REPEAT_COUNTER
            old_steer = 0

      # --------------Step 2-------------------------------
      steer_carla = steer_out / (max_steer_angle * STEER_RATIO * -1)

      steer_carla = np.clip(steer_carla, -1, 1)
      steer_out = steer_carla * (max_steer_angle * STEER_RATIO * -1)
      old_steer = steer_carla * (max_steer_angle * STEER_RATIO * -1)

      vc.throttle = throttle_out / 0.6
      vc.steer = steer_carla
      vc.brake = brake_out
      vehicle.apply_control(vc)

      # --------------Step 3-------------------------------
      vel = vehicle.get_velocity()
      speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # in m/s
      vehicle_state.speed = speed
      vehicle_state.vel = vel
      vehicle_state.angle = steer_out
      vehicle_state.cruise_button = cruise_button
      vehicle_state.is_engaged = is_openpilot_engaged

      if rk.frame % PRINT_DECIMATION == 0:
        print("frame: ", "engaged:", is_openpilot_engaged, "; throttle: ", round(vc.throttle, 3), "; steer(c/deg): ",
              round(vc.steer, 3), round(steer_out, 3), "; brake: ", round(vc.brake, 3))

      if rk.frame % 5 == 0:
        world.tick()
      rk.keep_time()
      self.started = True

  def close(self):
    self.started = False
    self._exit_event.set()
    # print('\ndestroying %d vehicles' % len(self._vehicles_list))
    # self._client.apply_batch([carla.command.DestroyActor(x) for x in self._vehicles_list])
    time.sleep(0.5)
    for s in self._carla_objects:
      try:
        s.destroy()
      except Exception as e:
        print("Failed to destroy carla object", e)
    for t in reversed(self._threads):
      t.join()

  def run(self, queue, retries=-1):
    bridge_p = Process(target=self.bridge_keep_alive, args=(queue, retries), daemon=True)
    bridge_p.start()
    return bridge_p


if __name__ == "__main__":
  q: Any = Queue()
  args = parse_args()

  carla_bridge = CarlaBridge(args)
  p = carla_bridge.run(q)

  if args.joystick:
    # start input poll for joystick
    from tools.sim.lib.manual_ctrl import wheel_poll_thread

    wheel_poll_thread(q)
  else:
    # start input poll for keyboard
    from tools.sim.lib.keyboard_ctrl import keyboard_poll_thread

    keyboard_poll_thread(q)
  p.join()
