from PIL import Image

from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation
from grid_adventure.step import Action
from grid_adventure.grid import step as grid_step
from grid_adventure.entities import *
from collections import deque
import heapq
from torchvision import datasets, transforms
import base64, io, torch
import zlib as _z; _decomp = _z.decompress

direction_map = {
            Action.UP:    (0, -1),
            Action.DOWN:  (0,  1),
            Action.LEFT:  (-1, 0),
            Action.RIGHT: (1,  0),
        }





class Agent:
    def __init__(self):
        self.nextActions = deque()
        self.wallPositions = set()
        self.gemPositions = set()
        self.keyPositions = set()
        self.doorPositions = set()
        self.unlockedDoorPositions = set()  
        # never account for adding unlocked doors to the doorPositions set, only add to unlockedDoorPositions when we see an opened door

        self.boxPositions = set()
        self.lavaPositions = set()
        self.coinPositions = set()
        self.shieldPositions = set()
        self.bootsPositions = set()
        self.ghostPositions = set()

        self.gridWidth = 0
        self.gridHeight = 0
        self.exitPosition = None
        self.agentPosition = None
        self.keysHeldByAgent = 0
        self.agentHealth = 5

        self.model = get_model()
        self.model.eval()
        self.classes = [
            'boots', 'box', 'coin', 'exit', 'floor',
            'gem', 'ghost', 'human', 'key', 'lava',
            'locked', 'opened', 'shield', 'wall'
        ]
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

    def initialise_env(self, state: GridState):
        self.gridWidth = state.width
        self.gridHeight = state.height

        for x in range(state.width):
            for y in range(state.height):
                for entity in state.objects_at((x, y)):
                    pos = (x, y)
                    if isinstance(entity, AgentEntity):
                        self.agentPosition = pos
                        self.agentHealth = entity.health.current_health
                        self.keysHeldByAgent = sum(1 for item in entity.inventory_list if isinstance(item, KeyEntity))
                    elif isinstance(entity, WallEntity):
                        self.wallPositions.add(pos)
                    elif isinstance(entity, GemEntity):
                        self.gemPositions.add(pos)
                    elif isinstance(entity, KeyEntity):
                        self.keyPositions.add(pos)
                    elif isinstance(entity, ExitEntity):
                        self.exitPosition = pos
                    elif isinstance(entity, LockedDoorEntity):
                        self.doorPositions.add(pos)
                    elif isinstance(entity, BoxEntity):
                        self.boxPositions.add(pos)
                    elif isinstance(entity, LavaEntity):
                        self.lavaPositions.add(pos)
                    elif isinstance(entity, CoinEntity):
                        self.coinPositions.add(pos)
                    elif isinstance(entity, ShieldPowerUpEntity):  
                        self.shieldPositions.add(pos)
                    elif isinstance(entity, SpeedPowerUpEntity):  
                        self.bootsPositions.add(pos)
                    elif isinstance(entity, PhasingPowerUpEntity): 
                        self.ghostPositions.add(pos)
                    elif isinstance(entity, UnlockedDoorEntity):
                        self.unlockedDoorPositions.add(pos)
                        self.doorPositions.add(pos) 

    def step(self, state: GridState | ImageObservation) -> Action:
        if not self.nextActions:
            if isinstance(state, dict):
                self.parseImage(state)
            else:
                self.initialise_env(state)
            self.plan()
        if self.nextActions:
            return self.nextActions.popleft()
        return Action.WAIT

    # envState: walls, lavas, doors positions are constant
        # gems, keys, coins, boots, shield, ghosts, opened doors, boxes
    # agentState: position, health, numKeys, boots, shield, ghosts
    def plan(self):
        gemPos = frozenset(self.gemPositions)
        keyPos = frozenset(self.keyPositions)
        coinPos = frozenset(self.coinPositions)
        bootsPos = frozenset(self.bootsPositions)
        shieldPos = frozenset(self.shieldPositions)
        ghostPos = frozenset(self.ghostPositions)
        openDoorPos = frozenset(self.unlockedDoorPositions)
        boxPos = frozenset(self.boxPositions)

        EnvState = (gemPos, keyPos, coinPos, bootsPos, shieldPos, ghostPos, openDoorPos, boxPos)
        AgentState = (self.agentPosition, self.agentHealth, self.keysHeldByAgent, 0, 0, 0)

        # parent, agent, env, actions, cost
        start = Node(None, AgentState, EnvState, None, 0)
        h = self.heuristic(self.agentPosition, gemPos, keyPos, self.keysHeldByAgent)

        PQ = []
        heapq.heappush(PQ, (h, start))
        visited = {start.visit_key()}
        goal_node = None

        while PQ:
            _, node = heapq.heappop(PQ)
            pos, remaining_gems = node.agentPos, node.EnvState[0]

            if pos == self.exitPosition and not remaining_gems:
                goal_node = node
                break

            for child in self.neighbours(node):
                hashable = child.visit_key()
                if hashable not in visited:
                    visited.add(hashable)
                    h = self.heuristic(child.agentPos, child.EnvState[0], child.EnvState[1], child.numKeys)
                    heapq.heappush(PQ, (child.cost + h, child))

        if goal_node:
            node = goal_node
            while node.parent is not None:
                self.nextActions.appendleft(node.action)
                node = node.parent
        else:
            self.nextActions.append(Action.WAIT)


    def neighbours(self, node):
        for action in Action:
            yield from self.constructNeighbour(node, action)
            
    def constructNeighbour(self, node, action):
        if action == Action.WAIT:
            return
        elif action == Action.PICK_UP:
            yield from self.executePickUp(node)
        elif action == Action.USE_KEY:
            yield from self.executeUseKey(node)
        else:
            yield from self.executeMove(node, action)

    def isValidNextPos(self, node, dx, dy):
        nextX, nextY = node.agentPos[0] + dx, node.agentPos[1] + dy

        if not (0 <= nextX < self.gridWidth and 0 <= nextY < self.gridHeight):
            return False
        if node.numGhost > 0:
            return True
        nextPos = (nextX, nextY)
        if nextPos in self.wallPositions:
            return False
        if nextPos in self.doorPositions and nextPos not in node.openDoorPos:
            return False
        if nextPos in node.boxPos:
            nextBoxX = nextX + dx
            nextBoxY = nextY + dy
            nextBoxPos = (nextBoxX, nextBoxY)
            if not (0 <= nextBoxX < self.gridWidth and 0 <= nextBoxY < self.gridHeight):
                return False
            if nextBoxPos in self.wallPositions:
                return False
            if nextBoxPos in self.lavaPositions:
                return False
            if nextBoxPos in self.doorPositions and nextBoxPos not in node.openDoorPos:
                return False
            if nextBoxPos in node.boxPos:
                return False
        return True

    def isValidBootsMove(self, pos, dx, dy, boxPos, openDoorPos, numGhost):
        nextX, nextY = pos[0] + dx, pos[1] + dy
        if not (0 <= nextX < self.gridWidth and 0 <= nextY < self.gridHeight):
            return False
        if numGhost > 0:
            return True
        nextPos = (nextX, nextY)
        if nextPos in self.wallPositions:
            return False
        if nextPos in self.doorPositions and nextPos not in openDoorPos:
            return False
        if nextPos in boxPos:
            bx, by = nextX + dx, nextY + dy
            if not (0 <= bx < self.gridWidth and 0 <= by < self.gridHeight):
                return False
            if (bx, by) in self.wallPositions or (bx, by) in self.lavaPositions:
                return False
            if (bx, by) in self.doorPositions and (bx, by) not in openDoorPos:
                return False
            if (bx, by) in boxPos:
                return False
        return True


    def executeMove(self, node, action):
        agentPos = node.agentPos
        dx, dy = direction_map.get(action)
        
        if not self.isValidNextPos(node, dx, dy):
            return
        
        nextX, nextY = agentPos[0] + dx, agentPos[1] + dy
        nextAgentPos = (nextX, nextY)
        nextHealth = node.agentHealth 
        nextNumBoots = node.numBoots
        nextNumGhost = node.numGhost
        nextNumShield = node.numShield

        nextBoxPos = node.boxPos

        if nextAgentPos in self.lavaPositions:
            if nextNumGhost > 0:
                pass
            elif nextNumShield > 0:
                nextNumShield = node.numShield -1 if node.numShield > 0 else 0
                pass
            else:
                nextHealth -= 2  
                if nextHealth <= 0:
                    return
        if nextAgentPos in nextBoxPos:
            if nextNumGhost > 0:
                pass
            else:
                next_box_x, next_box_y = nextAgentPos[0] + dx, nextAgentPos[1] + dy
                nextBoxPos = set(nextBoxPos)
                nextBoxPos.remove(nextAgentPos)
                nextBoxPos.add((next_box_x, next_box_y))
                nextBoxPos = frozenset(nextBoxPos)

        if (nextNumBoots > 0):
            withboots_next_pos = (nextAgentPos[0] + dx, nextAgentPos[1] + dy)

            if not self.isValidBootsMove(nextAgentPos, dx, dy, nextBoxPos, node.openDoorPos, nextNumGhost):
                pass
            else:
                if withboots_next_pos in self.lavaPositions:
                    if nextNumGhost > 0:
                        nextAgentPos = withboots_next_pos
                        pass
                    elif nextNumShield > 0:
                        nextNumShield = node.numShield -1 if node.numShield > 0 else 0
                        nextAgentPos = withboots_next_pos 
                        pass
                    else: 
                        if nextHealth - 2 <= 0:
                            pass  
                        else:
                            nextHealth -= 2
                            nextAgentPos = withboots_next_pos

                elif withboots_next_pos in nextBoxPos:
                    if nextNumGhost > 0:
                        pass
                    else:
                        next_box_x, next_box_y = withboots_next_pos[0] + dx, withboots_next_pos[1] + dy
                        nextBoxPos = set(nextBoxPos)
                        nextBoxPos.remove(withboots_next_pos)
                        nextBoxPos.add((next_box_x, next_box_y))
                        nextBoxPos = frozenset(nextBoxPos)
                    nextAgentPos = withboots_next_pos
                else:
                    nextAgentPos = withboots_next_pos
                
            
        nextNumBoots = max(node.numBoots - 1, 0)
        nextNumGhost = max(node.numGhost - 1, 0)

        new_state = (node.gemPos, node.keyPos, node.coinPos, node.bootsPos, node.shieldPos, node.ghostPos, node.openDoorPos, nextBoxPos)
        new_agent = (nextAgentPos, nextHealth, node.numKeys, nextNumBoots, nextNumShield, nextNumGhost)
        yield Node(node, new_agent, new_state, action, node.cost + 3)

    def executeUseKey(self, node):
        if node.numKeys <= 0:
            return
        NextnumBoots = max(node.numBoots - 1, 0)
        NextnumGhost = max(node.numGhost - 1, 0)
        for _, (dx, dy) in direction_map.items():
            adj = (node.agentPos[0] + dx, node.agentPos[1] + dy)
            if adj in self.doorPositions and adj not in node.openDoorPos:
                next_opened = node.openDoorPos | {adj}
                new_state = (node.gemPos, node.keyPos, node.coinPos, node.bootsPos, node.shieldPos, node.ghostPos, next_opened, node.boxPos)
                new_agent = (node.agentPos, node.agentHealth, node.numKeys - 1, NextnumBoots, node.numShield, NextnumGhost)
                yield Node(node, new_agent, new_state, Action.USE_KEY, node.cost + 3)
        
    def executePickUp(self, node):
        agentPos = node.agentPos
        
        if (not self.canCollect(agentPos, node.EnvState)):
            return
        
        NextnumBoots = max(node.numBoots - 1, 0)
        NextnumGhost = max(node.numGhost - 1, 0)
        numShield = node.numShield

        if agentPos in node.gemPos:
            new_gems = node.gemPos - {agentPos}
            new_state = (new_gems, node.keyPos, node.coinPos, node.bootsPos, node.shieldPos, node.ghostPos, node.openDoorPos, node.boxPos)
            new_agent = (agentPos, node.agentHealth, node.numKeys, NextnumBoots, node.numShield, NextnumGhost)
            yield Node(node, new_agent, new_state, Action.PICK_UP, node.cost + 3)
        elif agentPos in node.coinPos:
            new_coins = node.coinPos - {agentPos}
            new_state = (node.gemPos, node.keyPos, new_coins, node.bootsPos, node.shieldPos, node.ghostPos, node.openDoorPos, node.boxPos)
            new_agent = (agentPos, node.agentHealth, node.numKeys, NextnumBoots, node.numShield, NextnumGhost)
            yield Node(node, new_agent, new_state, Action.PICK_UP, node.cost - 2)
        elif agentPos in node.keyPos:
            if node.numKeys >= len(self.doorPositions) - len(node.openDoorPos):
                return
            new_keys = node.keyPos - {agentPos}
            new_state = (node.gemPos, new_keys, node.coinPos, node.bootsPos, node.shieldPos, node.ghostPos, node.openDoorPos, node.boxPos)
            new_agent = (agentPos, node.agentHealth, node.numKeys + 1, NextnumBoots, node.numShield, NextnumGhost)
            yield Node(node, new_agent, new_state, Action.PICK_UP, node.cost + 3)
        elif agentPos in node.shieldPos:
            if len(self.lavaPositions) == 0:
                return
            new_shields = node.shieldPos - {agentPos}
            new_state = (node.gemPos, node.keyPos, node.coinPos, node.bootsPos, new_shields, node.ghostPos, node.openDoorPos, node.boxPos)
            new_agent = (agentPos, node.agentHealth, node.numKeys, NextnumBoots, node.numShield + 5, NextnumGhost)
            yield Node(node, new_agent, new_state, Action.PICK_UP, node.cost + 3)
        elif agentPos in node.bootsPos:
            new_boots = node.bootsPos - {agentPos}
            new_state = (node.gemPos, node.keyPos, node.coinPos, new_boots, node.shieldPos, node.ghostPos, node.openDoorPos, node.boxPos)
            new_agent = (agentPos, node.agentHealth, node.numKeys, 5, node.numShield, NextnumGhost)
            yield Node(node, new_agent, new_state, Action.PICK_UP, node.cost + 3)
        elif agentPos in node.ghostPos:
            new_ghosts = node.ghostPos - {agentPos}
            new_state = (node.gemPos, node.keyPos, node.coinPos, node.bootsPos, node.shieldPos, new_ghosts, node.openDoorPos, node.boxPos)
            new_agent = (agentPos, node.agentHealth, node.numKeys, NextnumBoots, node.numShield, 5)
            yield Node(node, new_agent, new_state, Action.PICK_UP, node.cost + 3)



    def heuristic(self, pos, remaining_gems, floor_keys, keys_held):
        if pos is None or self.exitPosition is None:
            return 0

        if remaining_gems:
            furthest_gem = max(
                remaining_gems,
                key=lambda g: abs(g[0] - self.exitPosition[0]) + abs(g[1] - self.exitPosition[1])
            )
            return (abs(pos[0] - furthest_gem[0]) + abs(pos[1] - furthest_gem[1])) * 3
        return (abs(pos[0] - self.exitPosition[0]) + abs(pos[1] - self.exitPosition[1]))* 3





    # Utilities

    def canCollect(self, pos, state):
        remaining_gems = state[0]
        floor_keys = state[1]
        coins = state[2] 
        speed_powerups = state[3]
        shield_powerups = state[4]
        phasing_powerups = state[5]

        return (pos in remaining_gems or pos in floor_keys or pos in coins or
                pos in shield_powerups or pos in speed_powerups or pos in phasing_powerups)
    

# envState: walls, lavas, doors positions are constant
    # gems, keys, coins, boots, shield, ghosts, opened doors, boxes
# agentState: position, health, numKeys, boots, shield, ghosts
class Node:
    def __init__(self, parentNode, AgentState, EnvState, action, cost):
        self.agentPos = AgentState[0]
        self.agentHealth = AgentState[1]
        self.numKeys = AgentState[2]
        self.numBoots = AgentState[3]
        self.numShield = AgentState[4]
        self.numGhost = AgentState[5]

        self.gemPos = EnvState[0]
        self.keyPos = EnvState[1]
        self.coinPos = EnvState[2]
        self.bootsPos = EnvState[3]
        self.shieldPos = EnvState[4]
        self.ghostPos = EnvState[5]
        self.openDoorPos = EnvState[6]
        self.boxPos = EnvState[7]

        self.EnvState = EnvState
        self.parent = parentNode
        self.action = action
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost
    
    def visit_key(self):
        return (self.agentPos, self.numKeys, self.agentHealth,
                self.numBoots, self.numShield, self.numGhost, self.EnvState)