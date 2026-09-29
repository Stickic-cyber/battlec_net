import sys
from enum import Enum
from typing import NamedTuple

INT_SIZE = 64
PROTOCOL_MAJOR = 3
game: "Game" = None
ct: "Controller" = None

UINT64_MIN = 0
UINT64_MAX = (1 << INT_SIZE) - 1
UINT32_MAX = (1 << 32) - 1

class Constants:
    """What the engine is fixed at: 500 rounds, a 7x7 window, length 3 at spawn, length 2 minimum."""
    MAX_ROUNDS: int = 500
    VISION_RADIUS: int = 3
    VISION_SIZE: int = 2 * VISION_RADIUS + 1
    INITIAL_LENGTH: int = 3
    MIN_SIZE: int = 2

_OFFSET_BY_VALUE = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
# where a tile keeps the edge on that side
_INDEX_BY_VALUE = {"N": 0, "E": 1, "S": 2, "W": 3}

class Direction(Enum):
    """One of the four compass directions. North is up, and y grows downwards."""
    NORTH = "N"
    EAST = "E"
    SOUTH = "S"
    WEST = "W"

    __hash__ = object.__hash__
    def __init__(self, value: str) -> None:
        self._offset: tuple[int, int] = _OFFSET_BY_VALUE[value]
        self._index: int = _INDEX_BY_VALUE[value]
    @property
    def value(self) -> str:
        """The protocol letter, N, E, S or W."""
        return self._value_

    @classmethod
    def get_direction_list(cls) -> list["Direction"]:
        """A fresh list of all four, north, east, south, west."""
        return _DIRECTIONS[::]
        
    def get_offset(self) -> tuple[int, int]:
        """The (dx, dy) of one step this way, with y growing downwards."""
        return self._offset

    def get_opposite(self) -> "Direction":
        """The reverse direction."""
        return _OPPOSITE[self]

    def get_left(self) -> "Direction":
        """A quarter turn anticlockwise."""
        return _LEFT[self]

    def get_right(self) -> "Direction":
        """A quarter turn clockwise."""
        return _RIGHT[self]

_DIRECTIONS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
_DIRECTION_BY_VALUE = {d.value: d for d in _DIRECTIONS}
_OPPOSITE = {d: _DIRECTIONS[(i + 2) % 4] for i, d in enumerate(_DIRECTIONS)}
_LEFT = {d: _DIRECTIONS[(i + 3) % 4] for i, d in enumerate(_DIRECTIONS)}
_RIGHT = {d: _DIRECTIONS[(i + 1) % 4] for i, d in enumerate(_DIRECTIONS)}

class Team(Enum):
    """Which side a dragon plays for."""
    A = "A"
    B = "B"

    def get_enemy_team(self) -> "Team":
        """The other team."""
        return Team.A if self == Team.B else Team.B

_TEAM_BY_VALUE = {"A": Team.A, "B": Team.B}

class Position:
    """A tile's x and y, with (0, 0) at the top left and y growing downwards."""
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Position({self.x}, {self.y})"

    def add_dir(self, direction: Direction) -> "Position":
        """The position one step that way, wrapped around the map edges."""
        active_game = game
        if active_game is None:
            raise RuntimeError("init() must be called before using wrapped positions")
        dx, dy = direction._offset
        return Position((self.x + dx) % active_game.width,
                        (self.y + dy) % active_game.height)

    def is_in_map(self):
        """Whether these coordinates are on the board, which only matters for ones you worked out yourself."""
        return self.x >= 0 and self.y >= 0 and self.x < game.width and self.y < game.height
    
    def is_in_vision(self):
        """Whether this tile is inside your 7x7 window this turn."""
        if game is None:
            raise RuntimeError("init() must be called before checking vision")
        width, height = game.width, game.height
        if ct is None:
            raise RuntimeError("init() must be called before checking vision")
        pos = ct.get_position()
        if not (0 <= self.x < width and 0 <= self.y < height):
            return False
        left, top = pos.x - Constants.VISION_RADIUS, pos.y - Constants.VISION_RADIUS
        column = (self.x - left) % width
        row = (self.y - top) % height
        if column >= Constants.VISION_SIZE or row >= Constants.VISION_SIZE:
            return False
        return True

class Game:
    """The board and the round number, which every dragon sees the same way."""
    round_num: int = 0
    width: int = 0
    height: int = 0
    unit_limit: int = 0

    def __init__(self, width: int, height: int, unit_limit: int):
        self.round_num = 1
        self.width = width
        self.height = height
        self.unit_limit = unit_limit

    def get_round_num(self) -> int:
        """The round being played; the game ends after 500 of them."""
        return self.round_num

    def get_map_size(self) -> tuple[int, int]:
        """The map's (width, height) in tiles."""
        return self.width, self.height

    def get_unit_limit(self) -> int:
        """The most dragons one team may have alive at once, which is what caps splitting."""
        return self.unit_limit

class Vision:
    """The 7x7 block of tiles around your head, parsed as you ask for it."""
    __slots__ = ("_tile_lines", "_body_lines", "_edge_lines", "_head", "_game",
                 "_left", "_top", "_tiles", "_dragon_parts", "_horizontal_edges", "_vertical_edges")

    def __init__(self, tile_lines: list[str], body_lines: list[str], edge_lines: list[str],
                 head: "DragonPart", game_state: Game):
        self._tile_lines = tile_lines
        self._body_lines = body_lines
        self._edge_lines = edge_lines
        self._head = head
        self._game = game_state
        if tile_lines:
            self._left = head.position.x - Constants.VISION_RADIUS
            self._top = head.position.y - Constants.VISION_RADIUS
        self._tiles: list["Tile | None"] = [None] * len(tile_lines)
        self._dragon_parts: dict[tuple[int, int], "DragonPart"] = None

    def get_tiles(self) -> list["Tile"]:
        """All 49 tiles, row by row from the top left."""
        if None in self._tiles:
            for i in range(len(self._tiles)):
                self._tile(i)
        return self._tiles[::]

    def get_tile(self, pos: Position) -> "Tile | None":
        """The tile at pos, or None if pos is outside the window."""
        if not self._tiles:
            return None
        width, height = self._game.width, self._game.height
        if not (0 <= pos.x < width and 0 <= pos.y < height):
            return None
        column = (pos.x - self._left) % width
        row = (pos.y - self._top) % height
        if column >= Constants.VISION_SIZE or row >= Constants.VISION_SIZE:
            return None
        return self._tile(row * Constants.VISION_SIZE + column)

    def _tile(self, i: int) -> "Tile":
        tile = self._tiles[i]
        if tile is None:
            if self._dragon_parts is None:
                self._read_dragons_and_edges()
            x, y, has_pearl, pearl_time = self._tile_lines[i].split()
            x, y = int(x), int(y)
            position = Position(x, y)
            entity = self._dragon_parts.get((x, y))

            west = i + i // Constants.VISION_SIZE
            horizontal, vertical = self._horizontal_edges, self._vertical_edges
            tile = self._tiles[i] = Tile(position, entity, int(pearl_time), has_pearl == "1", (
                horizontal[i], vertical[west + 1], horizontal[i + Constants.VISION_SIZE], vertical[west]))
        return tile

    def _read_dragons_and_edges(self) -> None:
        head = self._head
        self._dragon_parts = dragon_parts = {}
        for line in self._body_lines:
            team, dragon_id, x, y, facing, is_dragon_head = line.split()
            dragon_id, x, y = int(dragon_id), int(x), int(y)
            if dragon_id == head.dragon_id and is_dragon_head == "1":
                dragon_parts[x, y] = head
            else:
                dragon_parts[x, y] = DragonPart(Position(x, y), dragon_id, _TEAM_BY_VALUE[team],
                                                _DIRECTION_BY_VALUE[facing], is_dragon_head == "1")

        # the horizontal edge rows come first, and there is one more of them than tile rows
        rows = Constants.VISION_SIZE + 1
        horizontal = "".join(self._edge_lines[:rows]).split()
        vertical = "".join(self._edge_lines[rows:]).split()
        self._horizontal_edges = [_HORIZONTAL_EDGES[text] for text in horizontal]
        self._vertical_edges = [_VERTICAL_EDGES[text] for text in vertical]

class Tile:
    """One square of the board, with whatever stands on it and its four edges."""
    __slots__ = ("position", "dragon_part", "pearl_time", "pearl", "_edges", "_edges_by_direction")

    def __init__(self, position: Position, dragon_part: "DragonPart", pearl_time: int, has_pearl: bool,
                 edges: tuple["Edge", "Edge", "Edge", "Edge"]):
        self.position = position
        self.dragon_part = dragon_part
        self.pearl_time = pearl_time
        self.pearl = has_pearl
        # north, east, south, west
        self._edges = edges
        self._edges_by_direction = None

    @property
    def edges(self) -> dict[Direction, "Edge"]:
        """The four edges keyed by direction."""
        if self._edges_by_direction is None:
            self._edges_by_direction = dict(zip(_DIRECTIONS, self._edges))
        return self._edges_by_direction

    def get_edge(self, direction: Direction) -> "Edge":
        """The edge you would cross stepping that way."""
        return self._edges[direction._index]

    def get_dragon(self) -> "DragonPart | None":
        """The dragon segment standing here, or None if the tile is clear."""
        return self.dragon_part

    def has_pearl(self) -> bool:
        """Whether a pearl is on this tile right now."""
        return self.pearl
        
    def get_pearl_time(self) -> int:
        """Rounds until this tile next tries to spawn a pearl, or -1 if it never does."""
        return self.pearl_time

    def get_position(self) -> Position:
        """This tile's position on the board."""
        return self.position

class DragonPart:
    """One segment of a dragon, head or body."""
    __slots__ = ("position", "dragon_id", "team", "dir", "is_dragon_head")

    def __init__(self, position: Position, dragon_id: int, team: Team,
                 direction: Direction, is_dragon_head: bool):
        self.position = position
        self.dragon_id = dragon_id
        self.team = team
        self.dir = direction
        self.is_dragon_head = is_dragon_head

    def __repr__(self) -> str:
        return f"DragonPart(id={self.dragon_id}, dir={self.dir})"

    def get_position(self) -> Position:
        """Where this segment is."""
        return self.position

    def get_id(self) -> int:
        """The dragon this segment belongs to."""
        return self.dragon_id

    def get_team(self) -> Team:
        """The team that dragon plays for."""
        return self.team

    def get_dir(self) -> Direction:
        """The direction this segment faces.
        A head faces where it is heading, and a body segment faces towards the head."""
        return self.dir

    def is_head(self) -> bool:
        """Whether this is the head; stepping into another dragon's head kills both of you."""
        return self.is_dragon_head

class EdgeType(Enum):
    """What lies between two tiles: EMPTY, KELP or PORTAL."""
    EMPTY = 0
    KELP = 1
    PORTAL = 2

class Edge:
    """The boundary between two tiles."""
    __slots__ = ("is_horizontal", "edge_type", "portal_id")

    def __init__(self, is_horizontal: bool, edge_type: EdgeType, portal_id: int = -1):
        self.is_horizontal = is_horizontal
        self.edge_type = edge_type
        self.portal_id = portal_id

    def is_passable(self) -> bool:
        """Whether a dragon can cross this edge; kelp is the only thing that stops one."""
        return self.edge_type != EdgeType.KELP

    def is_portal(self) -> bool:
        """Whether crossing this edge comes out at its partner edge."""
        return self.edge_type == EdgeType.PORTAL

    def get_edge_type(self) -> EdgeType:
        """Whether this edge is EMPTY, KELP or PORTAL."""
        return self.edge_type

    def get_portal_id(self) -> int:
        """The id this portal shares with its far edge, or -1 if this edge is not a portal."""
        return self.portal_id

class _Edges(dict):
    """Edge objects by their protocol text: "." empty, "w" kelp, a number for a portal id.
    Edges with the same text share one object, so treat them as read-only."""

    def __init__(self, is_horizontal: bool):
        self.is_horizontal = is_horizontal
        self["."] = Edge(is_horizontal, EdgeType.EMPTY)
        self["w"] = Edge(is_horizontal, EdgeType.KELP)

    def __missing__(self, text: str) -> Edge:
        edge = self[text] = Edge(self.is_horizontal, EdgeType.PORTAL, int(text))
        return edge

_HORIZONTAL_EDGES = _Edges(True)
_VERTICAL_EDGES = _Edges(False)

class SonarEchoes(NamedTuple):
    kelp: int = 0
    ally: int = 0
    ally_head: int = 0
    enemy: int = 0
    enemy_head: int = 0

class Controller:
    """Your dragon: what it sees, how long it is, and the commands it sends this turn."""
    length: int = 0
    unit_count: int = 0
    unit_limit: int = 0
    head: DragonPart = None
    vision: Vision = None

    def __init__(self, dragon_id: int, team: Team, direction: Direction,
                 vision: Vision, unit_limit: int, sonar_messages: list[int] = None):
        self.length = Constants.INITIAL_LENGTH
        self.unit_count = 1
        self.unit_limit = unit_limit
        self.head = DragonPart(None, dragon_id, team, direction, True)
        self.vision = vision
        self.sonar_messages = list(sonar_messages) if sonar_messages is not None else []
        self.sonar_echoes = SonarEchoes()

    def get_length(self) -> int:
        """Segments your dragon has, counting the head."""
        return self.length

    def get_unit_count(self) -> int:
        """Dragons your team has alive, this one included."""
        return self.unit_count

    def get_head(self) -> DragonPart:
        """Your dragon's head segment."""
        return self.head

    def get_id(self) -> int:
        """Your dragon's id, which is also its place in the turn order."""
        return self.head.dragon_id

    def get_team(self) -> Team:
        """The team you play for."""
        return self.head.team

    def get_dir(self) -> Direction:
        """The way your head points at the start of this turn."""
        return self.head.dir

    def get_vision(self) -> Vision:
        """The tiles around your head."""
        return self.vision

    def get_tiles(self) -> list[Tile]:
        """All 49 tiles in view, row by row from the top left."""
        return self.vision.get_tiles()

    def get_tile(self, pos: Position) -> "Tile | None":
        """The tile at pos, or None if pos is outside the window."""
        return self.vision.get_tile(pos)

    def get_position(self) -> Position:
        """Your head's position, already wrapped into the map."""
        return self.head.position

    def make_move(self, direction: Direction):
        """Steps one tile that way; of everything you send, the last action is the one applied."""
        print(f"MOVE {direction.value}")

    def make_moves(self, directions: list[Direction]):
        """Sprints one step per direction in the list.
        The helper sends whatever you pass, and n steps cost n - 1 segments, so your dragon must be longer than n."""
        moves = "".join(direction.value for direction in directions)
        print(f"MOVE {moves}")

    def can_split(self, child_size: int) -> bool:
        """Whether a child of that many segments is legal this turn."""
        return (Constants.MIN_SIZE <= child_size
                and self.length - child_size >= Constants.MIN_SIZE
                and self.unit_count < self.unit_limit)

    def do_split(self, child_size: int):
        """Splits that many segments off your tail.
        The child is those segments reversed, and it takes its own turn later in the same round."""
        print(f"SPLIT {child_size}")

    def output_log(self, *message):
        """Prints a line that shows against this turn in the replay.
        Every line costs points, so keep logging light: see the Timeouts page."""
        print("LOG", *message)

    def draw_indicator_dot(self, pos: Position, r: int, g: int, b: int):
        """Draws a dot on the replay's board, which changes nothing in the game."""
        print(f"DOT {pos.x} {pos.y} {r} {g} {b}")

    def draw_indicator_line(self, start: Position, end: Position, r: int, g: int, b: int):
        """Draws a line on the replay's board."""
        print(f"LINE {start.x} {start.y} {end.x} {end.y} {r} {g} {b}")

    def set_indicator_string(self, message: str):
        """Labels your dragon with this text for the turn."""
        print(f"INDICATOR {message}")

    def get_sonar_messages(self) -> list[int]:
        """Values that reached you since your last turn, in the order they were sent."""
        return self.sonar_messages[::]

    def get_sonar_echoes(self) -> SonarEchoes:
        return self.sonar_echoes

    def send_sonar(self, direction_or_message: Direction | int, message: int | None = None) -> bool:
        if message is None:
            return self._send_facing_sonar(direction_or_message)
        if not isinstance(direction_or_message, Direction) or not _is_uint64(message):
            return False
        print(f"SONAR {direction_or_message.value} {message}")
        return True

    def _send_facing_sonar(self, message: int) -> bool:
        if not _is_uint64(message) or message > UINT32_MAX:
            return False
        print(f"SONAR {message}")
        return True

def _is_uint64(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and UINT64_MIN <= value <= UINT64_MAX)

def init() -> tuple[Controller, Game]:
    """Reads the spawn block and returns your (controller, game), which stay valid all match."""
    global ct, game

    read = sys.stdin.readline
    dragon_id = int(read().split()[1])
    team = Team(read().split()[1])
    _, width, height = read().split()
    unit_limit = int(read().split()[1])

    game = Game(int(width), int(height), unit_limit)
    controller = Controller(dragon_id, team, Direction.NORTH, Vision([], [], [], None, game), unit_limit)
    ct = controller

    return controller, game

def update(controller: Controller, game_state: Game) -> bool:
    """Reads the next turn into the controller, and returns False once the game is over or your dragon has died."""
    read = sys.stdin.readline

    # judge ends each round input w blank line
    first = read()
    while first == "\n":
        first = read()
    if first == "":
        raise EOFError("unexpected end of controller input")
    if first.startswith("ENDGAME"):
        return False
    game_state.round_num = int(first.split()[1])

    head = controller.head
    head.dir = _DIRECTION_BY_VALUE[read().split()[1]]
    controller.length = int(read().split()[1])
    controller.unit_count = int(read().split()[1])
    num_sonar_msgs = int(read().split()[1])
    controller.sonar_messages = [int(read()) for _ in range(num_sonar_msgs)]

    # lazy parse vision
    # "x y hasPearl pearlIn" for each vision tile, row by row from the top left
    first_tile_line = read()
    controller.sonar_echoes = SonarEchoes()
    if first_tile_line.startswith("ECHOES"):
        controller.sonar_echoes = SonarEchoes(*map(int, first_tile_line.split()[1:]))
        first_tile_line = read()
    tile_lines = [first_tile_line] + [read() for _ in range(Constants.VISION_SIZE ** 2 - 1)]
    # "team dragonId x y facing isHead" for each dragon part in vision
    num_dragon_parts = int(read().split()[1])
    body_lines = [read() for _ in range(num_dragon_parts)]
    # VISION_SIZE + 1 rows of horizontal edges, then VISION_SIZE rows of vertical edges
    edge_lines = [read() for _ in range(2 * Constants.VISION_SIZE + 1)]

    # middle tile is head
    x, y, _, _ = tile_lines[len(tile_lines) // 2].split()
    head.position = Position(int(x), int(y))
    controller.vision = Vision(tile_lines, body_lines, edge_lines, head, game_state)

    return True

def end_turn():
    """Ends the turn and flushes everything you printed."""
    print(f"PROTOCOL {PROTOCOL_MAJOR}")
    print("ENDTURN", flush=True)
