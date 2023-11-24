import itertools        # iteration tools
import random
from typing import List

from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    Circle,
    CollisionCheckQuery,
    CollisionCheckResult,
    MapDefinition,
    PlacedPrimitive,
    Rectangle,
)

# _all_ is a special python variable used to define a list of names that should be 
#       exported when a module is imported using the 'from module import*'

_all_ = ["CollisionChecker"]


class CollisionChecker:
    params: MapDefinition       # Parameter of the class

    def init(self, context: Context):   # Constructor
        context.info("init()")          # Log message of the context 

    # on_received_set_params is called when the collision checker receives parameters
    def on_received_set_params(self, context: Context, data: MapDefinition):
        context.info("initialized")
        self.params = data
    # on_received_query is called when a collision checker query is received
    def on_received_query(self, context: Context, data: CollisionCheckQuery):
        collided = check_collision(
            environment=self.params.environment, robot_body=self.params.body, robot_pose=data.pose
        )
        result = CollisionCheckResult(collided)
        context.write("response", result)

def rototranslate_primitive(primitive: PlacedPrimitive, pose: FriendlyPose) -> PlacedPrimitive:
    new_pose = FriendlyPose(x=primitive.pose.x + pose.x, y=primitive.pose.y + pose.y, theta=primitive.pose.theta_deg + pose.theta_deg)
    return PlacedPrimitive(pose=new_pose, primitive=primitive.primitive, motion=primitive.motion, appearance=primitive.appearance)

def check_collision(
    environment: List[PlacedPrimitive], robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose
) -> bool:
    rototranslated_robot = [rototranslate_primitive(body, robot_pose) for body in robot_body]
    collided = check_collision_list(rototranslated_robot, environment)
    return bool(collided)


# check_collision_list checks for collisions between the robot and the enviroment objects
#    by iterating over all pairs and calling check_collision_shape()

def check_collision_list(
    rototranslated_robot: List[PlacedPrimitive], environment: List[PlacedPrimitive]
) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly
    for robot, envObject in itertools.product(rototranslated_robot, environment):
        if check_collision_shape(robot, envObject):
            return True
    return False

def check_collision_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    if isinstance(a.primitive, Circle) and isinstance(b.primitive, Circle):
        distance_squared = (a.pose.x - b.pose.x)*2 + (a.pose.y - b.pose.y)*2
        sum_radii_squared = (a.primitive.radius + b.primitive.radius)**2
        return bool(distance_squared <= sum_radii_squared)

    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Circle):
        closest_x = max(a.pose.x + a.primitive.xmin, min(b.pose.x, a.pose.x + a.primitive.xmax))
        closest_y = max(a.pose.y + a.primitive.ymin, min(b.pose.y, a.pose.y + a.primitive.ymax))
        distance_squared = (closest_x - b.pose.x)*2 + (closest_y - b.pose.y)*2
        return bool(distance_squared < b.primitive.radius**2)

    if isinstance(a.primitive, Rectangle) and isinstance(b.primitive, Rectangle):
        overlap_x = max(0, min(a.pose.x + a.primitive.xmax, b.pose.x + b.primitive.xmax) - max(a.pose.x + a.primitive.xmin, b.pose.x + b.primitive.xmin))
        overlap_y = max(0, min(a.pose.y + a.primitive.ymax, b.pose.y + b.primitive.ymax) - max(a.pose.y + a.primitive.ymin, b.pose.y + b.primitive.ymin))
        return bool(overlap_x > 0 and overlap_y > 0)

    # Default case: No collision (returning False)
    return False



















